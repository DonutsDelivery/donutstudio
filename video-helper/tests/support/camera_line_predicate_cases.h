#pragma once
#include "camera_ray_coefficient_cases.h"
#include "../../src/gpu_backend/fixture_native_camera_line_predicates.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace videohelper::tests::cameralinecases
{
namespace cl=arbitgpu::cameraline;
namespace cr=videohelper::tests::cameraraycases;
namespace bits=videohelper::tests::exactlinecases;
namespace cc=arbitgpu::cameraraycoefficients;
using Integer=bits::Integer;
using Line=std::array<Integer,3>;
// Independent dynamic binary-digit oracle, never production limb arithmetic.
// Incidence expands six determinant terms directly; event order divides into
// reduced unbounded rationals rather than using the native four-limb det kernel.
inline Integer number(std::int64_t n)
{
    auto a=cr::integer(static_cast<std::uint32_t>(n<0?-n:n));
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
    std::array<int,3> exponent{};int common=0;bool found=false;
    Line a;
    for(unsigned k=0;k<3;k++)if(values[k].numerator.sign){
        // Reduced dyadics have a power-of-two denominator. Canonicalize using
        // individual digits and division, independent of CC shifts/width scans.
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
        if(a[k].bits.size()>30)return false;
        storeInteger(at+1+9*k,a[k],8);
    }
    return true;
}
inline cc::Result modelRecord(const cc::Request& request)
{
    cc::Result r{};const auto model=cr::oracle(request);if(!model.valid||request[1]!=30)return r;
    for(unsigned k=0;k<17;k++)if(!storeForm(r.data()+61+28*k,model.forms[k]))return {};
    for(unsigned k=0;k<2;k++){
        std::array<std::uint32_t,28> scalar{};cr::Vec form{};form[0]=model.heights[k];
        if(!storeForm(scalar.data(),form))return {};
        std::copy_n(scalar.begin(),10,r.begin()+537+10*k);
    }
    r[0]=1;r[1]=cc::maximumVisits(30);r[2]=1;r[3]=30;std::copy(request.begin(),request.end(),r.begin()+4);return r;
}
inline cl::Request request(const cc::Request& parent,cl::Operation op,unsigned a,unsigned b,unsigned c=0,unsigned d=0)
{
    cl::Request r{};std::copy(parent.begin(),parent.end(),r.begin());r[57]=cl::kVersion;
    r[58]=static_cast<unsigned>(op);r[59]=cl::maximumScalarOperations(op);r[60]=a;r[61]=b;r[62]=c;r[63]=d;return r;
}
struct Case {const char* label;cl::ProbeInput input;};
inline Case native(const char* label,const cc::Request& parent,cl::Operation op,unsigned a,unsigned b,unsigned c=0,unsigned d=0)
{
    Case test{label,{}};const auto r=request(parent,op,a,b,c,d);std::copy(r.begin(),r.end(),test.input.begin());return test;
}
inline void setLine(cc::Result& r,unsigned index,std::int64_t a,std::int64_t b,std::int64_t c)
{
    auto* at=r.data()+61+28*index;std::fill_n(at,28,0u);
    cr::Vec v{{{number(a),cr::integer(1)},{number(b),cr::integer(1)},{number(c),cr::integer(1)}}};
    storeForm(at,v);
}
inline Case synthetic(const char* label,cl::Operation op,const std::array<std::array<std::int64_t,3>,4>& lines)
{
    const auto parent=cr::base();auto test=native(label,parent,op,0,1,op==cl::Operation::Cross?0:2,
        op==cl::Operation::CompareX||op==cl::Operation::CompareY?3:0);
    cc::Result r{};r[0]=1;r[1]=cc::maximumVisits(30);r[2]=1;r[3]=30;std::copy(parent.begin(),parent.end(),r.begin()+4);
    for(unsigned k=0;k<4;k++)setLine(r,k,lines[k][0],lines[k][1],lines[k][2]);
    r[538]=r[539]=r[548]=r[549]=1; // synthetic nonzero canonical H scalars
    // Synthetic records are mathematical fixtures, NEVER constructor evidence.
    test.input[65]=1;std::copy(r.begin(),r.end(),test.input.begin()+66);return test;
}
inline Line readLine(const cc::Result& r,unsigned index)
{
    Line line;
    for(unsigned k=0;k<3;k++){
        const auto at=62+28*index+9*k;line[k]=cr::integer(r[at+1]);
        if(r[at]==2)line[k]=bits::negative(line[k]);
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
    if(parent[0]!=1||parent[1]!=30||parent[2]<cc::maximumVisits(30)||parent[3]
        ||r[0]!=1||r[1]!=cc::maximumVisits(30)||r[2]==0||r[2]>r[1]||r[3]!=30)return false;
    if(!std::equal(parent.begin(),parent.end(),r.begin()+4))return false;
    // Independent canonical check: decode each full eight-word magnitude to
    // binary digits/rationals and require exact round-trip through the oracle's
    // reduction/normalization, not production canonical()/validateRecord().
    for(unsigned form=0;form<19;form++){
        const unsigned at=form<17?61+28*form:form==17?537:547,count=form<17?3:1;
        const auto exponent=r[at]<=0x7fffffffu?static_cast<std::int64_t>(r[at]):static_cast<std::int64_t>(r[at])-4294967296ll;
        if(exponent< -512||exponent>512)return false;
        cr::Vec values{};
        for(unsigned j=0;j<count;j++){
            const auto sign=r[at+1+9*j];if(sign>2)return false;
            Integer magnitude;for(unsigned bit=0;bit<256;bit++)magnitude.bits.push_back((r[at+2+9*j+bit/32]>>(bit%32))&1u);
            magnitude.sign=1;bits::trim(magnitude);
            if((magnitude.sign==0)!=(sign==0)||magnitude.bits.size()>30)return false;
            values[j]=cr::readScalar(r[at],sign,r.data()+at+2+9*j);
        }
        if(form>=17&&!values[0].numerator.sign)return false;
        std::array<std::uint32_t,28> normalized{};
        if(!storeForm(normalized.data(),values)||!std::equal(normalized.begin(),normalized.begin()+1+9*count,r.begin()+at))return false;
    }
    return true;
}
inline cl::Result oracle(const Case& test)
{
    cl::Request q{};std::copy_n(test.input.begin(),65,q.begin());cl::Result result{};
    std::copy(q.begin(),q.end(),result.begin()+34);
    const auto op=static_cast<cl::Operation>(q[58]);const auto charge=cl::maximumScalarOperations(op);result[1]=charge;
    if(q[57]!=1||q[58]>3||q[59]<charge||q[64]||q[60]>=17||q[61]>=17||q[62]>=17||q[63]>=17
        ||(q[58]==0&&(q[62]||q[63]))||(q[58]==1&&q[63]))return result;
    cc::Request parent{};std::copy_n(q.begin(),57,parent.begin());cc::Result r{};
    if(test.input[65]==0)r=modelRecord(parent);
    else if(test.input[65]==1)std::copy_n(test.input.begin()+66,557,r.begin());
    else return result;
    if(!oracleRecordAccepted(r,parent))return result;
    const auto a=readLine(r,q[60]),b=readLine(r,q[61]),c=readLine(r,q[62]),d=readLine(r,q[63]);
    const auto p=intersection(a,b);Line other{};const auto category=classification(a,b,p);int sign=0;
    if(op==cl::Operation::Incidence){
        if(category>=3)return result;
        std::array<Integer,12> words{};std::copy(a.begin(),a.end(),words.begin());
        std::copy(b.begin(),b.end(),words.begin()+3);std::copy(c.begin(),c.end(),words.begin()+6);
        auto value=bits::determinant(words);if(bits::det2(a[0],a[1],b[0],b[1]).sign<0)value=bits::negative(value);
        sign=value.sign;
    }
    if(op==cl::Operation::CompareX||op==cl::Operation::CompareY){
        other=intersection(c,d);if(category!=1||classification(c,d,other)!=1)return result;
        const unsigned axis=op==cl::Operation::CompareX?0:1;
        const auto lhs=cr::ratio({p[axis],cr::integer(1)},{p[2],cr::integer(1)});
        const auto rhs=cr::ratio({other[axis],cr::integer(1)},{other[2],cr::integer(1)});
        sign=cr::subtract(lhs,rhs).numerator.sign;
    }
    result[0]=1;result[2]=category;result[3]=bits::signWord(sign);
    for(unsigned k=0;k<3;k++){
        storeInteger(result.data()+4+5*k,p[k],4);storeInteger(result.data()+19+5*k,other[k],4);
    }
    return result;
}
inline std::vector<Case> cases()
{
    using O=cl::Operation;const auto parent=cr::base();
    std::vector<Case> tests{
        native("native constructed closed shared endpoint",parent,O::Incidence,4,5,12),
        native("native constructed opposed seam coincidence",parent,O::Cross,4,12),
        native("native constructed equal-depth zero form",parent,O::Cross,16,4),
        native("native constructed finite cross",parent,O::Cross,4,5),
        native("native constructed ordered endpoint x",parent,O::CompareX,4,5,4,6),
        native("native constructed ordered endpoint y",parent,O::CompareY,4,5,4,6),
        native("native constructed equal event",parent,O::CompareX,4,5,4,5)};
    auto reversed=parent;
    for(unsigned k=0;k<3;k++)std::swap(reversed[31+k],reversed[34+k]);
    std::swap(reversed[28],reversed[29]);
    tests.push_back(native("native constructed reversed winding",reversed,O::CompareY,4,5,4,6));
    const std::array<std::array<std::int64_t,3>,4> square{{{1,0,-1},{0,1,-1},{1,0,-2},{0,1,-2}}};
    tests.push_back(synthetic("synthetic rational x order",O::CompareX,square));
    tests.push_back(synthetic("synthetic rational y order",O::CompareY,square));
    auto lines=square;lines[2]={1,-1,0};tests.push_back(synthetic("synthetic closed seam incidence",O::Incidence,lines));
    lines=square;lines[2]={1,-1,1};tests.push_back(synthetic("synthetic displaced seam incidence",O::Incidence,lines));
    lines=square;lines[1]={1,0,-2};tests.push_back(synthetic("synthetic parallel",O::Cross,lines));
    lines=square;lines[1]={-1,0,1};tests.push_back(synthetic("synthetic coincident opposed signs",O::Cross,lines));
    lines=square;lines[0]={0,0,0};tests.push_back(synthetic("synthetic zero form distinct from coincidence",O::Cross,lines));
    tests.push_back(synthetic("synthetic zero intersection incidence refuses",O::Incidence,lines));
    lines=square;lines[1]={0,-1,1};tests.push_back(synthetic("synthetic negative denominator normalizes",O::CompareX,lines));
    lines=square;lines[2]=lines[0];lines[3]=lines[1];tests.push_back(synthetic("synthetic exact equal events",O::CompareY,lines));
    constexpr std::int64_t m=1073741823;
    lines={{{m,m,-m},{m,-m,m-1},{-m,m,m},{m,m-1,-m}}};
    tests.push_back(synthetic("synthetic extreme 30-bit cross carry",O::Cross,lines));
    tests.push_back(synthetic("synthetic extreme 30-bit incidence cancellation",O::Incidence,lines));
    tests.push_back(synthetic("synthetic extreme 30-bit event carry",O::CompareX,lines));
    tests.push_back(synthetic("synthetic extreme 30-bit event y",O::CompareY,lines));
    lines={{{m,m-1,-m},{m-1,m-2,-m+1},{m,m-2,-m},{m-1,m-3,-m+1}}};
    tests.push_back(synthetic("synthetic one integer unit cancellation",O::CompareX,lines));
    auto scaled=synthetic("synthetic positive dyadic scale ignored",O::CompareX,square);
    scaled.input[66+61]=static_cast<std::uint32_t>(-512);scaled.input[66+89]=512;tests.push_back(scaled);
    auto reject=synthetic("synthetic forged upper limb refuses",O::Cross,square);reject.input[66+64]=1;tests.push_back(reject);
    reject=synthetic("synthetic full 32-bit magnitude refuses",O::Cross,square);reject.input[66+63]=0x40000000u;tests.push_back(reject);
    reject=synthetic("synthetic noncanonical signed zero refuses",O::Cross,square);reject.input[66+71]=1;tests.push_back(reject);
    reject=synthetic("synthetic out-of-domain exponent refuses",O::Cross,square);reject.input[66+61]=513;tests.push_back(reject);
    reject=synthetic("synthetic all-even noncanonical line refuses",O::Cross,square);reject.input[66+63]=2;reject.input[66+81]=2;tests.push_back(reject);
    reject=synthetic("synthetic forged request echo refuses",O::Cross,square);reject.input[66+8]^=1;tests.push_back(reject);
    reject=synthetic("synthetic wrong constructed status refuses",O::Cross,square);reject.input[66]=0;tests.push_back(reject);
    reject=synthetic("synthetic wrong precision refuses",O::Cross,square);reject.input[66+3]=256;tests.push_back(reject);
    reject=synthetic("synthetic forged constructor work refuses",O::Cross,square);reject.input[66+2]=cc::maximumVisits(30)+1;tests.push_back(reject);
    reject=synthetic("synthetic unknown operation refuses",O::Cross,square);reject.input[58]=4;tests.push_back(reject);
    reject=synthetic("synthetic invalid line index refuses",O::Cross,square);reject.input[60]=17;tests.push_back(reject);
    reject=synthetic("synthetic version refuses",O::Cross,square);reject.input[57]=2;tests.push_back(reject);
    reject=synthetic("synthetic budget one operation short refuses",O::CompareX,square);--reject.input[59];tests.push_back(reject);
    reject=synthetic("synthetic zero budget full charge retained",O::Cross,square);reject.input[59]=0;tests.push_back(reject);
    lines=square;lines[1]={1,0,-2};tests.push_back(synthetic("synthetic infinite event ordering refuses",O::CompareX,lines));
    reject=synthetic("synthetic constructor version refuses",O::Cross,square);reject.input[0]=reject.input[66+4]=2;tests.push_back(reject);
    reject=synthetic("synthetic zero constructor performed work refuses",O::Cross,square);reject.input[66+2]=0;tests.push_back(reject);
    reject=synthetic("synthetic zero plane height refuses",O::Cross,square);reject.input[66+538]=reject.input[66+539]=0;tests.push_back(reject);
    return tests;
}
inline bool compare(const Case& test,const cl::Result& actual,std::string& error)
{
    const auto expected=oracle(test);
    if(actual==expected)return true;
    for(unsigned k=0;k<cl::kResultWords;k++)if(actual[k]!=expected[k]){
        error=std::string(test.label)+" word "+std::to_string(k)+" actual "+std::to_string(actual[k])+" expected "+std::to_string(expected[k]);break;}
    return false;
}
// Test-local whole dispatch admission; constructor work/scratch are additional
// for mode0. Synthetic arithmetic buffers retain no constructor authority.
constexpr std::uint64_t kMaximumProbeWork=cl::maximumScalarOperations(cl::Operation::CompareX)+cc::maximumScalarOperations(30);
constexpr std::uint64_t kProbeScratchBytes=cl::kLogicalScratchBytes+cc::kLogicalScratchBytes;
inline std::uint64_t work(const Case& test)
{return cl::maximumScalarOperations(static_cast<cl::Operation>(test.input[58]))+
    (test.input[65]==0?cc::maximumScalarOperations(30):0);}
} // namespace videohelper::tests::cameralinecases
