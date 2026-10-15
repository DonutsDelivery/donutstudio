#pragma once
#include "../../src/gpu_backend/fixture_native_exact_line_predicates.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <initializer_list>
#include <vector>

namespace videohelper::tests::exactlinecases
{
namespace el=arbitgpu::exactline;
// Independent oracle: dynamically sized individual binary digits, shift/add
// multiplication, and six-term determinant expansion. It shares no shader limb,
// carry, capacity, Laplace expansion or work-counter implementation. No floats.
struct Integer { int sign=0;std::vector<std::uint8_t> bits; };
inline void trim(Integer& a){while(!a.bits.empty()&&!a.bits.back())a.bits.pop_back();if(a.bits.empty())a.sign=0;}
inline int magnitude(const Integer& a,const Integer& b)
{
    if(a.bits.size()!=b.bits.size())return a.bits.size()>b.bits.size()?1:-1;
    for(std::size_t k=a.bits.size();k>0;--k)if(a.bits[k-1]!=b.bits[k-1])return a.bits[k-1]>b.bits[k-1]?1:-1;
    return 0;
}
inline Integer add(const Integer& a,const Integer& b)
{
    if(!a.sign)return b;
    if(!b.sign)return a;
    Integer r;r.bits.resize(std::max(a.bits.size(),b.bits.size())+1);
    if(a.sign==b.sign){r.sign=a.sign;int carry=0;
        for(std::size_t k=0;k<r.bits.size();k++){int n=carry+(k<a.bits.size()?a.bits[k]:0)+(k<b.bits.size()?b.bits[k]:0);r.bits[k]=n%2;carry=n/2;}}
    else {const bool first=magnitude(a,b)>=0;const auto& x=first?a:b;const auto& y=first?b:a;r.sign=x.sign;int borrow=0;
        for(std::size_t k=0;k<r.bits.size();k++){int n=(k<x.bits.size()?x.bits[k]:0)-(k<y.bits.size()?y.bits[k]:0)-borrow;borrow=n<0;r.bits[k]=static_cast<std::uint8_t>(n<0?n+2:n);}}
    trim(r);return r;
}
inline Integer negative(Integer a){a.sign=-a.sign;return a;}
inline Integer multiply(const Integer& a,const Integer& b)
{
    Integer r;if(!a.sign||!b.sign)return r;
    for(std::size_t k=0;k<b.bits.size();k++)if(b.bits[k]){Integer shifted=a;shifted.sign=1;shifted.bits.insert(shifted.bits.begin(),k,0);r=add(r,shifted);}
    r.sign=a.sign*b.sign;return r;
}
inline Integer decode(std::uint32_t word)
{
    Integer r;const auto exponent=(word>>23)&255u,frac=word&0x7fffffu;
    const auto significand=exponent?frac|0x800000u:frac;
    if(!significand)return r;
    r.sign=word>>31?-1:1;
    r.bits.resize(exponent?exponent-1:0);
    for(unsigned k=0;k<24;k++)r.bits.push_back((significand>>k)&1u);
    trim(r);return r;
}
inline Integer det2(const Integer& a,const Integer& b,const Integer& c,const Integer& d)
{return add(multiply(a,d),negative(multiply(b,c)));}
inline std::array<Integer,3> cross(const std::array<Integer,12>& a,unsigned at)
{return {det2(a[at+1],a[at+2],a[at+4],a[at+5]),det2(a[at+2],a[at],a[at+5],a[at+3]),det2(a[at],a[at+1],a[at+3],a[at+4])};}
inline Integer determinant(const std::array<Integer,12>& a)
{
    Integer r;const unsigned terms[6][3]{{0,4,8},{1,5,6},{2,3,7},{2,4,6},{1,3,8},{0,5,7}};
    for(unsigned k=0;k<6;k++){auto t=multiply(multiply(a[terms[k][0]],a[terms[k][1]]),a[terms[k][2]]);r=add(r,k<3?t:negative(t));}return r;
}
inline std::uint32_t signWord(int sign){return sign>0?1u:sign<0?2u:0u;}
inline el::Result oracle(const el::Request& input)
{
    el::Result output{};const auto op=static_cast<el::Operation>(input[0]);const auto bound=el::maximumVisits(op);
    if(!bound||input[1]<bound||!input[2]||input[2]>35||input[3])return output;
    output[3]=bound;const unsigned count=op==el::Operation::Det2?4:op==el::Operation::Cross?6:
        op==el::Operation::Det3||op==el::Operation::Incidence?9:12;
    std::array<Integer,12> a;
    for(unsigned k=0;k<12;k++){if(k>=count){if(input[k+4])return output;continue;}
        if(((input[k+4]>>23)&255u)==255u)return output;
        a[k]=decode(input[k+4]);if(a[k].bits.size()>32*input[2])return output;}
    // Fixtures exercise reduced capacity only with an input that already cannot
    // fit, or admission refusal. The oracle doesn't emulate transient capacities.
    if(op!=el::Operation::Det2&&op!=el::Operation::Det3)
        for(unsigned k=0;k<count;k+=3)if(!a[k].sign&&!a[k+1].sign&&!a[k+2].sign)return output;
    Integer result;std::array<Integer,3> p{};
    unsigned classification=0;
    if(op==el::Operation::Det2)result=det2(a[0],a[1],a[2],a[3]);
    else {p=cross(a,0);classification=p[2].sign?1u:p[0].sign||p[1].sign?2u:3u;
        if(op==el::Operation::Cross){}
        else if(op==el::Operation::Det3||op==el::Operation::Incidence){if(op==el::Operation::Incidence&&classification==3)return output;result=determinant(a);}
        else {const auto q=cross(a,6);if(!p[2].sign||!q[2].sign)return output;
            const unsigned axis=op==el::Operation::CompareX?0:1;
            result=add(multiply(p[axis],q[2]),negative(multiply(q[axis],p[2])));
            if(p[2].sign*q[2].sign<0)result=negative(result);}}
    if(result.bits.size()>32*input[2])return output;
    output[0]=1;output[1]=signWord(result.sign);output[2]=op==el::Operation::Det3?0:classification;
    if(op==el::Operation::Cross)for(unsigned axis=0;axis<3;axis++){
        output[5+19*axis]=signWord(p[axis].sign);
        for(std::size_t bit=0;bit<p[axis].bits.size();bit++)if(p[axis].bits[bit])output[6+19*axis+bit/32]|=1u<<(bit%32);}
    return output;
}
struct Case {const char* label;el::Request input;};
inline el::Request request(el::Operation op,std::initializer_list<std::uint32_t> words)
{
    el::Request r{};r[0]=static_cast<std::uint32_t>(op);r[1]=el::maximumVisits(op);r[2]=35;
    std::copy(words.begin(),words.end(),r.begin()+4);return r;
}
inline std::vector<Case> cases()
{
    constexpr std::uint32_t z=0,nz=0x80000000u,o=0x3f800000u,n=0xbf800000u,t=0x40000000u,
        tiny=1u,large=0x7f7fffffu,minusLarge=0xff7fffffu;
    using O=el::Operation;
    std::vector<Case> c{
        {"det2 exact cancellation",request(O::Det2,{o,o,o,o})},
        {"det2 one ULP",request(O::Det2,{o,o,o,o+1})},
        {"det2 swapped sign",request(O::Det2,{o,o+1,o,o})},
        {"det2 signed zero",request(O::Det2,{nz,z,nz,z})},
        {"det2 worst product carry and signed sum",request(O::Det2,{large,large,minusLarge,large})},
        {"det2 subnormal cancellation",request(O::Det2,{tiny,tiny,tiny,tiny})},
        {"det3 exact cancellation",request(O::Det3,{large,large,large,large,large,large,large,large,large})},
        {"det3 six worst-size terms",request(O::Det3,{large,large,minusLarge,minusLarge,large,large,large,minusLarge,large})},
        {"det3 near cancellation",request(O::Det3,{o,o,o,o,o+1,o,o,o,o+1})},
        {"det3 swapped orientation",request(O::Det3,{o,o+1,o,o,o,o,o,o,o+1})},
        {"cross finite",request(O::Cross,{o,z,n,z,o,n})},
        {"cross parallel",request(O::Cross,{o,z,n,o,z,z})},
        {"cross coincident",request(O::Cross,{o,z,n,t,z,0xc0000000u})},
        {"zero line refuses",request(O::Cross,{z,nz,z,o,z,n})},
        {"incidence exact seam",request(O::Incidence,{o,z,n,z,o,n,o,n,z})},
        {"incidence one ULP",request(O::Incidence,{o,z,n,z,o,n,o,n,tiny})},
        {"incidence degenerate point refuses",request(O::Incidence,{o,z,n,o,z,n,z,o,n})},
        {"ordered x events",request(O::CompareX,{o,z,n,z,o,n,o,z,0xc0000000u,z,o,n})},
        {"equal events have no tie",request(O::CompareX,{o,z,n,z,o,n,z,o,n,o,z,n})},
        {"denominator reversal",request(O::CompareX,{z,o,n,o,z,n,o,z,0xc0000000u,z,o,n})},
        {"ordered y events",request(O::CompareY,{o,z,n,z,o,n,o,z,n,z,o,0xc0000000u})},
        {"infinite event refuses",request(O::CompareX,{o,z,n,o,z,z,o,z,n,z,o,n})},
        {"worst cross comparison",request(O::CompareX,{large,large,minusLarge,large,minusLarge,large,large,large,large,minusLarge,large,minusLarge})},
        {"subnormal finite events",request(O::CompareY,{tiny,z,tiny,z,tiny,tiny,tiny,z,tiny,z,tiny,tiny+1})},
        {"NaN refuses",request(O::Det2,{0x7fc00001u,o,o,o})},
        {"positive infinity refuses",request(O::Det3,{0x7f800000u,o,o,o,o,o,o,o,o})},
        {"negative infinity refuses",request(O::Cross,{0xff800000u,z,n,z,o,n})}
    };
    c.push_back({"long subtraction borrow",request(O::Det2,{large,tiny,tiny,large})});
    c.push_back({"worst comparison exponent spread",request(O::CompareX,{large,tiny,large,tiny,large,tiny,large,tiny,tiny,tiny,large,large})});
    auto intermediate=request(O::Det2,{large,tiny,tiny,large});intermediate[2]=9;
    c.push_back({"product capacity overflow refuses",intermediate});
    auto raisedCapacity=request(O::Det2,{o,o,o,o});raisedCapacity[2]=36;
    c.push_back({"capacity cannot exceed derived maximum",raisedCapacity});
    auto raisedWork=request(O::CompareX,{o,z,n,z,o,n,o,z,n,z,o,n});raisedWork[1]=0xffffffffu;
    c.push_back({"raised caller work limit does not raise native envelope",raisedWork});
    c.push_back({"reversed event order",request(O::CompareX,{o,z,0xc0000000u,z,o,n,o,z,n,z,o,n})});
    c.push_back({"signaling NaN refuses",request(O::Det2,{0x7f800001u,o,o,o})});
    c.push_back({"negative NaN refuses",request(O::Cross,{0xffc00001u,z,n,z,o,n})});
    // Every finite exponent, both signs; mantissa boundary carries, including
    // largest subnormal/minimum normal and maximum finite are compared exactly.
    for(std::uint32_t e=0;e<255;e++){
        const auto hi=(e<<23)|0x7fffffu,lo=e<<23;
        c.push_back({"all finite exponents",request(O::Det2,{hi,lo,lo,hi})});
        c.push_back({"all negative finite exponents",request(O::Det2,{hi|nz,hi,hi,hi|nz})});}
    auto shortWork=request(O::CompareX,{o,z,n,z,o,n,o,z,n,z,o,n});shortWork[1]--;c.push_back({"work admission refuses",shortWork});
    auto cap=request(O::Det2,{large,o,o,o});cap[2]=1;c.push_back({"capacity refuses before exact output",cap});
    auto reserved=request(O::Cross,{o,z,n,z,o,n});reserved[3]=1;c.push_back({"reserved request refuses",reserved});
    auto unused=request(O::Det2,{o,o,o,o});unused[15]=1;c.push_back({"unused word refuses",unused});
    auto unknown=request(O::Det2,{o,o,o,o});unknown[0]=7;c.push_back({"unsupported operation refuses",unknown});
    return c;
}
inline bool compare(const Case& test,const el::Result& actual,std::string& error)
{
    const auto expected=oracle(test.input);
    for(std::size_t k=0;k<actual.size();k++)if(k!=4&&actual[k]!=expected[k]){
        error=std::string(test.label)+": native/oracle word "+std::to_string(k)+" differs";return false;}
    if(actual[4]>actual[3]){error=std::string(test.label)+": native exceeded admitted limb visits";return false;}
    return true;
}
}
