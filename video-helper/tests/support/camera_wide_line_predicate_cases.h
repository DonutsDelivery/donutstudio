#pragma once
#include "camera_ray_coefficient_cases.h"
#include "../../src/gpu_backend/fixture_native_camera_wide_line_predicates.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace videohelper::tests::camerawidelinecases
{
namespace cw=arbitgpu::camerawideline;
namespace cr=videohelper::tests::cameraraycases;
namespace bits=videohelper::tests::exactlinecases;
namespace cc=arbitgpu::cameraraycoefficients;
using Integer=bits::Integer;
using Line=std::array<Integer,3>;
// Independent dynamic binary digits and reduced rationals, no EL arithmetic.
// Camera model uses independent rational rotations/six-term determinants;
// incidence expands six terms, and event order compares reduced fractions.
inline Integer number(std::int32_t n)
{
    auto a=cr::integer(static_cast<std::uint32_t>(n<0?-static_cast<std::int64_t>(n):n));
    if(n<0)a=bits::negative(a);
    return a;
}
inline void storeInteger(std::uint32_t* at,const Integer& a,unsigned limbs)
{
    at[0]=bits::signWord(a.sign);
    for(std::size_t k=0;k<a.bits.size();k++)if(a.bits[k]&&k/32<limbs)at[1+k/32]|=1u<<(k%32);
}
inline bool storeForm(std::uint32_t* at,const cr::Vec& values)
{
    std::array<int,3> exponent{};int common=0;bool found=false;Line a;
    for(unsigned k=0;k<3;k++)if(values[k].numerator.sign){
        const auto& denominator=values[k].denominator;
        if(denominator.bits.empty())return false;
        for(std::size_t bit=0;bit+1<denominator.bits.size();bit++)if(denominator.bits[bit])return false;
        exponent[k]=1-static_cast<int>(denominator.bits.size());a[k]=values[k].numerator;
        while(!a[k].bits.empty()&&!a[k].bits.front()){a[k].bits.erase(a[k].bits.begin());++exponent[k];}
        if(!found||exponent[k]<common)common=exponent[k];
        found=true;
    }
    if(common< -512||common>512)return false;
    at[0]=static_cast<std::uint32_t>(common);
    for(unsigned k=0;k<3;k++){
        if(a[k].sign)a[k].bits.insert(a[k].bits.begin(),exponent[k]-common,0);
        if(a[k].bits.size()>256)return false;
        storeInteger(at+1+9*k,a[k],8);
    }
    return true;
}
inline cc::Result modelRecord(const cc::Request& q)
{
    cc::Result r{};const auto model=cr::oracle(q);
    if(!model.valid||q[1]!=256)return r;
    for(unsigned k=0;k<17;k++)if(!storeForm(r.data()+61+28*k,model.forms[k]))return {};
    for(unsigned k=0;k<2;k++){
        std::array<std::uint32_t,28> scalar{};cr::Vec form{};form[0]=model.heights[k];
        if(!storeForm(scalar.data(),form))return {};
        std::copy_n(scalar.begin(),10,r.begin()+537+10*k);
    }
    r[0]=1;r[1]=cc::maximumVisits(256);r[2]=1;r[3]=256;std::copy(q.begin(),q.end(),r.begin()+4);return r;
}
inline cw::Request request(const cc::Request& parent,cw::Operation op,unsigned a,unsigned b,unsigned c=0,unsigned d=0)
{
    cw::Request r{};std::copy(parent.begin(),parent.end(),r.begin());r[57]=cw::kVersion;r[58]=static_cast<unsigned>(op);
    r[59]=cw::maximumScalarOperations(op);r[60]=a;r[61]=b;r[62]=c;r[63]=d;return r;
}
struct Case {const char* label;cw::ProbeInput input;bool constructorSupported=true;};
inline Case native(const char* label,const cc::Request& parent,cw::Operation op,unsigned a,unsigned b,
    unsigned c=0,unsigned d=0,bool constructorSupported=true)
{
    Case test{label,{},constructorSupported};const auto q=request(parent,op,a,b,c,d);
    std::copy(q.begin(),q.end(),test.input.begin());return test;
}
inline cc::Request sourceCamera(std::uint32_t precision=256)
{
    auto q=cr::base(precision);q[12]=0x3f800000u;q[14]=q[16]=0x3f3504f3u;q[17]=0x3ed413cdu;
    q[19]=0x3dcccccdu;q[20]=0x42c80000u;
    q[21]=q[22]=0x42000000u;q[8]=16;q[9]=13;
    // Source-grounded mathematical encodings of GL/Metal terminal fixture
    // camera=(0,0,1), q=(0,float(sqrt(.5)),0,float(sqrt(.5))), tangent of half
    // authored FOV0.7853981634, near0.1/far100,32x32 extent, receiver at x=-4.
    // Source backend uses std::tan(FOV*.5f). These nominated raw encodings are
    // NOT authenticated live uniform capture or the full T16/T11 emitted scene.
    // Two receiver triangles here measure arithmetic only; emitters/competitors,
    // raster ownership and actual ALL-triangle native parent remain absent.
    const std::uint32_t p[18]{0xc0800000u,0xbf800000u,0xc1000000u,0xc0800000u,0x3f800000u,0xc1000000u,
        0xc0800000u,0x3f800000u,0x40c00000u,0xc0800000u,0xbf800000u,0xc1000000u,
        0xc0800000u,0x3f800000u,0x40c00000u,0xc0800000u,0xbf800000u,0x40c00000u};
    for(unsigned tri=0;tri<2;tri++)for(unsigned k=0;k<9;k++)q[31+17*tri+k]=p[9*tri+k];
    q[28]=7;q[29]=8;q[30]=9;q[45]=7;q[46]=9;q[47]=10;return q;
}
inline Case synthetic(const char* label,cw::Operation op,const std::array<Line,4>& lines)
{
    const auto parent=cr::base(256);auto test=native(label,parent,op,0,1,op==cw::Operation::Cross?0:2,
        op==cw::Operation::CompareX||op==cw::Operation::CompareY?3:0);
    cc::Result r{};r[0]=1;r[1]=cc::maximumVisits(256);r[2]=1;r[3]=256;std::copy(parent.begin(),parent.end(),r.begin()+4);
    for(unsigned k=0;k<4;k++){
        cr::Vec values;for(unsigned j=0;j<3;j++)values[j]={lines[k][j],cr::integer(1)};
        storeForm(r.data()+61+28*k,values);
    }
    r[538]=r[539]=r[548]=r[549]=1;test.input[65]=1;std::copy(r.begin(),r.end(),test.input.begin()+66);return test;
}
inline Line readLine(const cc::Result& r,unsigned index)
{
    Line line;
    for(unsigned j=0;j<3;j++){
        const unsigned at=62+28*index+9*j;auto& a=line[j];a.sign=r[at]==1?1:r[at]==2?-1:0;
        for(unsigned bit=0;bit<256;bit++)a.bits.push_back((r[at+1+bit/32]>>(bit%32))&1u);
        bits::trim(a);
    }
    return line;
}
inline Line intersection(const Line& a,const Line& b)
{
    Line p{bits::det2(a[1],a[2],b[1],b[2]),bits::det2(a[2],a[0],b[2],b[0]),bits::det2(a[0],a[1],b[0],b[1])};
    if(p[2].sign<0){for(auto& v:p)v=bits::negative(v);}
    return p;
}
inline unsigned classification(const Line& a,const Line& b,const Line& p)
{
    if((!a[0].sign&&!a[1].sign&&!a[2].sign)||(!b[0].sign&&!b[1].sign&&!b[2].sign))return 4;
    return p[2].sign?1:p[0].sign||p[1].sign?2:3;
}
inline bool oracleRecordAccepted(const cc::Result& r,const cc::Request& parent)
{
    if(parent[0]!=1||parent[1]!=256||parent[2]<cc::maximumVisits(256)||parent[3]
        ||r[0]!=1||r[1]!=cc::maximumVisits(256)||r[2]==0||r[2]>r[1]||r[3]!=256)return false;
    if(!std::equal(parent.begin(),parent.end(),r.begin()+4))return false;
    // Canonical wide record roundtrip with independent dyadic reduction.
    for(unsigned form=0;form<19;form++){
        const unsigned at=form<17?61+28*form:form==17?537:547,count=form<17?3:1;
        const auto exponent=r[at]<=0x7fffffffu?static_cast<std::int64_t>(r[at]):static_cast<std::int64_t>(r[at])-4294967296ll;
        if(exponent< -512||exponent>512)return false;
        cr::Vec values{};
        for(unsigned j=0;j<count;j++){
            const auto sign=r[at+1+9*j];if(sign>2)return false;
            Integer magnitude;
            for(unsigned bit=0;bit<256;bit++)magnitude.bits.push_back((r[at+2+9*j+bit/32]>>(bit%32))&1u);
            magnitude.sign=1;bits::trim(magnitude);
            if((magnitude.sign==0)!=(sign==0))return false;
            values[j]=cr::readScalar(r[at],sign,r.data()+at+2+9*j);
        }
        if(form>=17&&!values[0].numerator.sign)return false;
        std::array<std::uint32_t,28> normalized{};
        if(!storeForm(normalized.data(),values)||!std::equal(normalized.begin(),normalized.begin()+1+9*count,r.begin()+at))return false;
    }
    return true;
}
inline cw::Result oracle(const Case& test)
{
    cw::Request q{};std::copy_n(test.input.begin(),65,q.begin());cw::Result result{};std::copy(q.begin(),q.end(),result.begin()+112);
    const auto op=static_cast<cw::Operation>(q[58]);const auto charge=cw::maximumScalarOperations(op);result[1]=charge;
    if(q[57]!=1||q[58]>3||q[59]<charge||q[64]||q[60]>=17||q[61]>=17||q[62]>=17||q[63]>=17
        ||(q[58]==0&&(q[62]||q[63]))||(q[58]==1&&q[63]))return result;
    cc::Request parent{};std::copy_n(q.begin(),57,parent.begin());cc::Result r{};
    if(test.input[65]==0){if(test.constructorSupported)r=modelRecord(parent);}
    else if(test.input[65]==1)std::copy_n(test.input.begin()+66,557,r.begin());
    else return result;
    if(!oracleRecordAccepted(r,parent))return result;
    const auto a=readLine(r,q[60]),b=readLine(r,q[61]),c=readLine(r,q[62]),d=readLine(r,q[63]);
    const auto p=intersection(a,b);Line other{};const auto category=classification(a,b,p);int sign=0;
    if(op==cw::Operation::Incidence){
        if(category>=3)return result;
        std::array<Integer,12> words{};std::copy(a.begin(),a.end(),words.begin());
        std::copy(b.begin(),b.end(),words.begin()+3);std::copy(c.begin(),c.end(),words.begin()+6);
        auto value=bits::determinant(words);
        if(bits::det2(a[0],a[1],b[0],b[1]).sign<0)value=bits::negative(value);
        sign=value.sign;
    }
    if(op==cw::Operation::CompareX||op==cw::Operation::CompareY){
        other=intersection(c,d);if(category!=1||classification(c,d,other)!=1)return result;
        const unsigned axis=op==cw::Operation::CompareX?0:1;
        const auto lhs=cr::ratio({p[axis],cr::integer(1)},{p[2],cr::integer(1)});
        const auto rhs=cr::ratio({other[axis],cr::integer(1)},{other[2],cr::integer(1)});
        sign=cr::subtract(lhs,rhs).numerator.sign;
    }
    result[0]=1;result[2]=category;result[3]=bits::signWord(sign);
    for(unsigned k=0;k<3;k++){storeInteger(result.data()+4+18*k,p[k],17);storeInteger(result.data()+58+18*k,other[k],17);}
    return result;
}
inline std::vector<Case> cases()
{
    using O=cw::Operation;const auto source=sourceCamera();
    std::vector<Case> tests{
        native("source sqrt-half camera precision30 constructor refuses",sourceCamera(30),O::Cross,6,10,0,0,false),
        native("source sqrt-half camera precision256 opposed seam",source,O::Cross,6,10),
        native("source sqrt-half camera closed endpoint",source,O::Incidence,4,6,10),
        native("source sqrt-half camera equal depth zero",source,O::Cross,16,4),
        native("source sqrt-half camera ordered event x",source,O::CompareX,4,6,4,5),
        native("source sqrt-half camera ordered event y",source,O::CompareY,4,6,4,5),
        native("source sqrt-half camera equal event",source,O::CompareX,4,6,4,6)};
    auto reversed=source;
    for(unsigned k=0;k<3;k++)std::swap(reversed[31+k],reversed[34+k]);
    std::swap(reversed[28],reversed[29]);tests.push_back(native("source reversed winding and denominator",reversed,O::CompareY,4,6,4,5));
    const auto original=cr::cases();
    tests.push_back(native("non-f32 generated coefficients",original[5].input,O::Cross,4,5));
    tests.push_back(native("non-power-of-two captured aspect and extents",original[6].input,O::CompareY,4,5,4,6));
    const std::array<Line,4> square{{{number(1),number(0),number(-1)},{number(0),number(1),number(-1)},
        {number(1),number(0),number(-2)},{number(0),number(1),number(-2)}}};
    tests.push_back(synthetic("synthetic rational x order",O::CompareX,square));
    tests.push_back(synthetic("synthetic rational y order",O::CompareY,square));
    auto lines=square;lines[2]={number(1),number(-1),number(0)};tests.push_back(synthetic("synthetic closed seam incidence",O::Incidence,lines));
    lines=square;lines[2]={number(1),number(-1),number(-1)};tests.push_back(synthetic("synthetic negative incidence",O::Incidence,lines));
    lines=square;lines[1]={number(1),number(0),number(-2)};tests.push_back(synthetic("synthetic parallel",O::Cross,lines));
    lines=square;lines[1]={number(-1),number(0),number(1)};tests.push_back(synthetic("synthetic opposed coincidence",O::Cross,lines));
    lines=square;lines[0]={};tests.push_back(synthetic("synthetic zero form",O::Cross,lines));
    lines=square;lines[1]={number(0),number(-1),number(1)};tests.push_back(synthetic("synthetic negative denominator order",O::CompareX,lines));
    lines=square;lines[2]=lines[0];lines[3]=lines[1];tests.push_back(synthetic("synthetic equal events",O::CompareY,lines));
    Integer m;m.sign=1;m.bits.assign(256,1);const auto almost=bits::add(m,number(-1)),minus=bits::negative(m);
    lines={{{m,m,minus},{m,minus,almost},{minus,m,m},{m,almost,minus}}};
    tests.push_back(synthetic("synthetic all eight limbs cross carry",O::Cross,lines));
    tests.push_back(synthetic("synthetic extreme256 incidence cancellation",O::Incidence,lines));
    tests.push_back(synthetic("synthetic extreme256 event carry x",O::CompareX,lines));
    tests.push_back(synthetic("synthetic extreme256 event carry y",O::CompareY,lines));
    lines[1]=lines[0];tests.push_back(synthetic("synthetic extreme256 exact cancellation",O::Cross,lines));
    auto scaled=synthetic("synthetic positive scale limits",O::CompareX,square);
    scaled.input[66+61]=static_cast<std::uint32_t>(-512);scaled.input[66+89]=512;tests.push_back(scaled);
    auto high=synthetic("synthetic high limb accepted rather than truncated",O::Cross,square);
    high.input[66+70]=0x80000000u;tests.push_back(high);
    auto reject=synthetic("synthetic invalid sign refuses",O::Cross,square);reject.input[66+62]=3;tests.push_back(reject);
    reject=synthetic("synthetic sign zero with high limb refuses",O::Cross,square);reject.input[66+62]=0;reject.input[66+70]=1;tests.push_back(reject);
    reject=synthetic("synthetic signed zero refuses",O::Cross,square);reject.input[66+71]=1;tests.push_back(reject);
    reject=synthetic("synthetic exponent outside domain refuses",O::Cross,square);reject.input[66+61]=513;tests.push_back(reject);
    reject=synthetic("synthetic all-even line refuses",O::Cross,square);reject.input[66+63]=reject.input[66+81]=2;tests.push_back(reject);
    reject=synthetic("synthetic forged source echo refuses",O::Cross,square);reject.input[66+8]^=1;tests.push_back(reject);
    reject=synthetic("synthetic wrong status refuses",O::Cross,square);reject.input[66]=0;tests.push_back(reject);
    reject=synthetic("synthetic wrong precision refuses",O::Cross,square);reject.input[66+3]=30;tests.push_back(reject);
    reject=synthetic("synthetic constructor version refuses",O::Cross,square);reject.input[0]=reject.input[66+4]=2;tests.push_back(reject);
    reject=synthetic("synthetic performed work beyond admission refuses",O::Cross,square);reject.input[66+2]=cc::maximumVisits(256)+1;tests.push_back(reject);
    reject=synthetic("synthetic zero performed work refuses",O::Cross,square);reject.input[66+2]=0;tests.push_back(reject);
    reject=synthetic("synthetic admitted constructor work refuses",O::Cross,square);--reject.input[66+1];tests.push_back(reject);
    reject=synthetic("synthetic zero plane height refuses",O::Cross,square);reject.input[66+538]=reject.input[66+539]=0;tests.push_back(reject);
    reject=synthetic("synthetic predicate version refuses",O::Cross,square);reject.input[57]=2;tests.push_back(reject);
    reject=synthetic("synthetic operation refuses",O::Cross,square);reject.input[58]=4;tests.push_back(reject);
    reject=synthetic("synthetic line index refuses",O::Cross,square);reject.input[60]=17;tests.push_back(reject);
    reject=synthetic("synthetic full work one short refuses",O::CompareX,square);--reject.input[59];tests.push_back(reject);
    reject=synthetic("synthetic zero work keeps full charge",O::Cross,square);reject.input[59]=0;tests.push_back(reject);
    lines=square;lines[1]=lines[0];tests.push_back(synthetic("synthetic coincident incidence refuses",O::Incidence,lines));
    lines=square;lines[1]={number(1),number(0),number(-2)};tests.push_back(synthetic("synthetic infinite ordering refuses",O::CompareX,lines));
    auto invalid=source;invalid[17]=0x7f800000u;tests.push_back(native("native nonfinite constructor refuses",invalid,O::Cross,4,5,0,0,false));
    invalid=source;invalid[13]=0x7f7fffffu;tests.push_back(native("native exponent precision overflow refuses",invalid,O::Cross,4,5,0,0,false));
    invalid=source;--invalid[2];tests.push_back(native("native constructor insufficient admission refuses",invalid,O::Cross,4,5,0,0,false));
    return tests;
}
inline bool compare(const Case& test,const cw::ProbeResult& actual,std::string& error)
{
    cw::Result predicate{};std::copy_n(actual.begin(),177,predicate.begin());const auto expected=oracle(test);
    if(predicate!=expected){
        for(unsigned k=0;k<177;k++)if(predicate[k]!=expected[k]){
            error=std::string(test.label)+" predicate word "+std::to_string(k)+" actual "+std::to_string(predicate[k])+" expected "+std::to_string(expected[k]);break;}
        return false;
    }
    cc::Result record{};std::copy_n(actual.begin()+177,557,record.begin());
    if(test.input[65]==0){
        cc::Request parent{};std::copy_n(test.input.begin(),57,parent.begin());
        return cr::compare({test.label,parent,test.constructorSupported},record,error);
    }
    if(std::equal(record.begin(),record.end(),test.input.begin()+66))return true;
    error=std::string(test.label)+" synthetic record diagnostic changed";return false;
}
constexpr std::uint64_t kMaximumProbeWork=cw::maximumScalarOperations(cw::Operation::CompareX)+
    cc::maximumScalarOperations(256)+cw::kProbeDiagnosticScalarOperations;
constexpr std::uint64_t kProbeScratchBytes=cw::kLogicalScratchBytes+cc::kLogicalScratchBytes;
inline std::uint64_t work(const Case& test)
{return cw::maximumScalarOperations(static_cast<cw::Operation>(test.input[58]))+cw::kProbeDiagnosticScalarOperations+
    (test.input[65]==0?cc::maximumScalarOperations(test.input[1]):0);}
} // namespace videohelper::tests::camerawidelinecases
