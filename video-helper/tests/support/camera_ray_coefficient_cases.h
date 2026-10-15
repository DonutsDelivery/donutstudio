#pragma once
#include "exact_line_predicate_cases.h"
#include "../../src/gpu_backend/fixture_native_camera_ray_coefficients.h"
#include <array>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

namespace videohelper::tests::cameraraycases
{
namespace cc=arbitgpu::cameraraycoefficients;
namespace bits=videohelper::tests::exactlinecases;
using Integer=bits::Integer;
// Independent unbounded rational oracle. Numerator/denominator are individual
// binary digits, with shift/add multiplication and binary long division/GCD.
// Camera rotation uses the cross-product expression on rational vectors; native
// construction expands the camera matrix and clears extent divisions first.
// Triangle/edge signs use the six-term determinant directly, not native cross/dot.
inline Integer integer(std::uint32_t n){Integer r;r.sign=n?1:0;while(n){r.bits.push_back(n&1u);n>>=1;}return r;}
inline Integer absolute(Integer a){if(a.sign){a.sign=1;}return a;}
inline std::pair<Integer,Integer> divide(Integer a,const Integer& divisor)
{
    Integer quotient,remainder;quotient.bits.resize(a.bits.size());quotient.sign=1;
    for(std::size_t k=a.bits.size();k>0;--k){remainder.bits.insert(remainder.bits.begin(),a.bits[k-1]);remainder.sign=1;bits::trim(remainder);
        if(bits::magnitude(remainder,divisor)>=0){remainder=bits::add(remainder,bits::negative(divisor));quotient.bits[k-1]=1;}}
    bits::trim(quotient);return {quotient,remainder};
}
inline Integer gcd(Integer a,Integer b){a=absolute(a);b=absolute(b);while(b.sign){auto r=divide(a,b).second;a=b;b=r;}return a;}
struct Rational {Integer numerator,denominator=integer(1);};
inline Rational reduce(Rational a)
{
    if(!a.numerator.sign)return {};
    const auto factor=gcd(a.numerator,a.denominator);const int sign=a.numerator.sign;
    a.numerator=divide(absolute(a.numerator),factor).first;a.numerator.sign=sign;
    a.denominator=divide(a.denominator,factor).first;return a;
}
inline Rational number(std::uint32_t n){return {integer(n),integer(1)};}
inline Rational negative(Rational a){a.numerator=bits::negative(a.numerator);return a;}
inline Rational add(const Rational& a,const Rational& b)
{return reduce({bits::add(bits::multiply(a.numerator,b.denominator),bits::multiply(b.numerator,a.denominator)),bits::multiply(a.denominator,b.denominator)});}
inline Rational subtract(const Rational& a,const Rational& b){return add(a,negative(b));}
inline Rational multiply(const Rational& a,const Rational& b)
{return reduce({bits::multiply(a.numerator,b.numerator),bits::multiply(a.denominator,b.denominator)});}
inline Rational ratio(const Rational& a,const Rational& b)
{
    auto numerator=bits::multiply(a.numerator,b.denominator);
    if(b.numerator.sign<0)numerator=bits::negative(numerator);
    return reduce({numerator,bits::multiply(a.denominator,absolute(b.numerator))});
}
inline bool equal(const Rational& a,const Rational& b){return subtract(a,b).numerator.sign==0;}
inline Rational decode(std::uint32_t word)
{
    const auto e=(word>>23)&255u,sig=(word&0x7fffffu)|(e?0x800000u:0u);
    Rational r=number(sig);if(!sig)return r;
    if(word>>31)r=negative(r);
    const int exponent=e?static_cast<int>(e)-150:-149;
    if(exponent>=0)r.numerator.bits.insert(r.numerator.bits.begin(),exponent,0);
    else r.denominator.bits.insert(r.denominator.bits.begin(),-exponent,0);
    return reduce(r);
}
using Vec=std::array<Rational,3>;
inline Vec subtract(const Vec& a,const Vec& b){return {subtract(a[0],b[0]),subtract(a[1],b[1]),subtract(a[2],b[2])};}
inline Vec cross(const Vec& a,const Vec& b)
{return {subtract(multiply(a[1],b[2]),multiply(a[2],b[1])),subtract(multiply(a[2],b[0]),multiply(a[0],b[2])),subtract(multiply(a[0],b[1]),multiply(a[1],b[0]))};}
inline Vec rotate(const std::array<Rational,4>& q,const Vec& p)
{
    Vec v{q[0],q[1],q[2]},inner=cross(v,p);
    for(unsigned k=0;k<3;k++)inner[k]=add(inner[k],multiply(q[3],p[k]));
    auto outer=cross(v,inner);Vec r;for(unsigned k=0;k<3;k++)r[k]=add(p[k],multiply(number(2),outer[k]));
    return r;
}
inline Rational determinant(const Vec& a,const Vec& b,const Vec& c)
{
    const Rational terms[]{multiply(multiply(a[0],b[1]),c[2]),multiply(multiply(a[1],b[2]),c[0]),
        multiply(multiply(a[2],b[0]),c[1]),multiply(multiply(a[2],b[1]),c[0]),
        multiply(multiply(a[1],b[0]),c[2]),multiply(multiply(a[0],b[2]),c[1])};
    Rational r;for(unsigned k=0;k<6;k++)r=add(r,k<3?terms[k]:negative(terms[k]));
    return r;
}
struct Model {std::array<Vec,17> forms;std::array<Rational,2> heights;bool valid=false;};
inline Model oracle(const cc::Request& request)
{
    Model model;
    const unsigned rawRanges[][2]{{10,23},{31,40},{48,57}};
    for(const auto& range:rawRanges)for(unsigned k=range[0];k<range[1];k++)if(((request[k]>>23)&255u)==255u)return model;
    Vec camera{decode(request[10]),decode(request[11]),decode(request[12])};
    std::array<Rational,4> q,proj;for(unsigned k=0;k<4;k++){q[k]=decode(request[13+k]);proj[k]=decode(request[17+k]);}
    auto width=decode(request[21]),height=decode(request[22]);
    if(width.numerator.sign<=0||height.numerator.sign<=0||proj[0].numerator.sign<=0||proj[1].numerator.sign<=0
        ||proj[2].numerator.sign<=0||subtract(proj[3],proj[2]).numerator.sign<=0
        ||subtract(width,number(request[8])).numerator.sign<=0||subtract(height,number(request[9])).numerator.sign<=0)return model;
    // Form the actual rational film coordinates with divisions; multiply by
    // positive width*height only AFTER independent quaternion evaluation.
    const auto scale=multiply(width,height),tAspect=multiply(proj[0],proj[1]);
    Vec film[3]{};
    film[0][0]=ratio(multiply(number(2),tAspect),width);
    film[1][1]=negative(ratio(multiply(number(2),proj[0]),height));
    film[2][0]=multiply(subtract(ratio(number(2*request[8]),width),number(1)),tAspect);
    film[2][1]=multiply(subtract(number(1),ratio(number(2*request[9]),height)),proj[0]);
    film[2][2]=negative(number(1));
    Vec d[3];for(unsigned k=0;k<3;k++){d[k]=rotate(q,film[k]);for(auto& component:d[k])component=multiply(component,scale);}
    if(determinant(d[0],d[1],d[2]).numerator.sign==0)return model;
    auto conjugate=q;for(unsigned k=0;k<3;k++)conjugate[k]=negative(conjugate[k]);
    Vec gd;
    for(unsigned k=0;k<3;k++)gd[k]=negative(rotate(conjugate,d[k])[2]);
    for(unsigned axis=0;axis<3;axis++)for(unsigned k=0;k<3;k++)model.forms[axis][k]=d[k][axis];
    model.forms[3]=gd;
    for(unsigned tri=0;tri<2;tri++){
        Vec vertex[3],relative[3];for(unsigned v=0;v<3;v++)for(unsigned k=0;k<3;k++)vertex[v][k]=decode(request[31+tri*17+v*3+k]);
        for(unsigned v=0;v<3;v++)relative[v]=subtract(vertex[v],camera);
        const auto e1=subtract(vertex[1],vertex[0]),e2=subtract(vertex[2],vertex[0]);
        model.heights[tri]=determinant(relative[0],relative[1],relative[2]);
        if(model.heights[tri].numerator.sign==0)return model;
        for(unsigned k=0;k<3;k++){
            for(unsigned edge=0;edge<3;edge++)model.forms[4+tri*6+edge][k]=determinant(relative[edge],relative[(edge+1)%3],d[k]);
            auto n=determinant(e1,e2,d[k]);model.forms[7+tri*6][k]=n;
            auto numerator=multiply(model.heights[tri],gd[k]);
            model.forms[8+tri*6][k]=subtract(numerator,multiply(proj[2],n));
            model.forms[9+tri*6][k]=subtract(numerator,multiply(proj[3],n));
        }
    }
    for(unsigned k=0;k<3;k++)model.forms[16][k]=subtract(multiply(model.heights[0],model.forms[13][k]),multiply(model.heights[1],model.forms[7][k]));
    model.valid=true;return model;
}
inline Rational readScalar(std::uint32_t exponent,std::uint32_t sign,const std::uint32_t* limbs)
{
    Rational r;r.numerator.sign=sign==1?1:sign==2?-1:0;
    for(unsigned k=0;k<8;k++)for(unsigned bit=0;bit<32;bit++)r.numerator.bits.push_back((limbs[k]>>bit)&1u);
    bits::trim(r.numerator);
    const std::int64_t e=exponent<=0x7fffffffu?exponent:static_cast<std::int64_t>(exponent)-4294967296ll;
    if(e>=0)r.numerator.bits.insert(r.numerator.bits.begin(),static_cast<std::size_t>(e),0);
    else r.denominator.bits.insert(r.denominator.bits.begin(),static_cast<std::size_t>(-e),0);
    return reduce(r);
}
// A mathematically valid rational model can still lie outside the constructor
// intermediate exponent/precision domain. Refusal cases claim no native proof.
struct Case {const char* label;cc::Request input;bool supported=true;};
inline cc::Request base(std::uint32_t precision=30)
{
    cc::Request r{};r[0]=1;r[1]=precision;r[2]=cc::maximumVisits(precision);
    r[4]=0xf1234567u;r[5]=0xe3456789u;r[6]=0x87654321u;r[7]=1;r[8]=1;r[9]=0;
    r[16]=0x3f800000u;r[17]=0x3f800000u;r[18]=0x3f800000u;r[19]=0x3e800000u;r[20]=0x41200000u;
    r[21]=r[22]=0x40000000u;
    // Same-owner square at z=-2. AB of triangle0 equals reversed CA of triangle1.
    const std::uint32_t positions[18]{0xbf800000u,0xbf800000u,0xc0000000u,0x3f800000u,0x3f800000u,0xc0000000u,0xbf800000u,0x3f800000u,0xc0000000u,
        0xbf800000u,0xbf800000u,0xc0000000u,0x3f800000u,0xbf800000u,0xc0000000u,0x3f800000u,0x3f800000u,0xc0000000u};
    for(unsigned tri=0;tri<2;tri++){
        r[23+17*tri]=tri;r[24+17*tri]=11;r[25+17*tri]=13;r[26+17*tri]=17;r[27+17*tri]=101+tri;
        r[28+17*tri]=7;r[29+17*tri]=tri?10:8;r[30+17*tri]=tri?8:9;
        for(unsigned k=0;k<9;k++)r[31+17*tri+k]=positions[9*tri+k];}
    return r;
}
inline std::vector<Case> cases()
{
    std::vector<Case> c{{"same-owner exact seam",base()}};
    auto r=base();for(unsigned k=0;k<17;k++)r[40+k]=r[23+k];
    c.push_back({"one triangle repeated and exact equal depth",r});
    r=base();r[50]=0xbfc00000u;r[53]=0xc0200000u;r[56]=0xc0000000u;c.push_back({"genuine depth exchange",r});
    auto reversed=r;for(unsigned k=0;k<3;k++)std::swap(reversed[31+k],reversed[34+k]);
    std::swap(reversed[28],reversed[29]);c.push_back({"reversed winding and denominator",reversed});
    r=base(256);r[13]=0x3f000000u;r[16]=0x3f800000u;c.push_back({"non-unit quaternion projector",r});
    r=base(256);r[17]=0x3f800001u;r[18]=0x3f800001u;c.push_back({"generated coefficient not binary32",r});
    r=base(256);r[18]=0x3fa66666u;r[21]=0x40400000u;r[22]=0x40a00000u;r[8]=1;r[9]=2;c.push_back({"non-power-of-two aspect and extents",r});
    r=base();r[19]=0x40000000u;r[20]=0x40400000u;c.push_back({"projector near plane exact equality",r});
    r=base();r[20]=0x40000000u;c.push_back({"projector far plane exact equality",r});
    r=base();r[31]=0x40000000u;r[34]=0x40000000u;r[37]=0x3f800000u;r[32]=0xbf800000u;r[35]=0x3f800000u;r[38]=0x3f800000u;r[33]=0xc0000000u;r[36]=0xc0000000u;r[39]=0;c.push_back({"genuine horizon form",r});
    r=base(256);r[50]=r[53]=r[56]=0xc0000001u;c.push_back({"one ULP plane separation",r});
    r=base();r[10]=0x80000000u;r[11]=0x80000000u;r[13]=0x80000000u;c.push_back({"canonical signed zero",r});
    r=base(256);for(unsigned tri=0;tri<2;tri++)for(unsigned k=0;k<9;k++){auto& word=r[31+17*tri+k];word=(word&0x80000000u)|1u;}c.push_back({"subnormal triangle intermediate exponent refuses",r,false});
    r=base(256);r[10]=1u;c.push_back({"supported subnormal camera displacement",r});
    r=base();for(unsigned k=0;k<3;k++)r[37+k]=r[34+k];
    c.push_back({"degenerate triangle refuses",r,false});
    r=base();r[33]=r[36]=r[39]=0;c.push_back({"plane through camera refuses",r,false});
    r=base();r[13]=r[14]=0x3f000000u;r[16]=0;c.push_back({"singular non-unit ray map refuses",r,false});
    r=base();r[17]=0x3f800001u;r[18]=0x3f800001u;c.push_back({"30-bit product precision refuses",r,false});
    r=base(256);r[17]=r[18]=r[21]=r[22]=1u;r[8]=r[9]=0;c.push_back({"exponent domain refuses",r,false});
    r=base(256);r[13]=0x7f7fffffu;c.push_back({"maximum finite precision refuses",r,false});
    r=base(256);r[13]=1u;c.push_back({"subnormal quaternion alignment refuses",r,false});
    r=base();r[17]=0x7f800000u;c.push_back({"infinite capture refuses",r,false});
    r=base();r[48]=0x7fc00001u;c.push_back({"NaN frozen position refuses",r,false});
    r=base();r[2]--;c.push_back({"insufficient work refuses before construction",r,false});
    r=base();r[1]=32;c.push_back({"unsupported precision refuses",r,false});
    r=base();r[8]=16777217u;c.push_back({"pixel integer conversion domain refuses",r,false});
    r=base(256);r[14]=r[16]=0x3f3504f3u;c.push_back({"rounded near-unit quaternion",r});
    r=base();r[3]=1;c.push_back({"reserved protocol word refuses",r,false});
    r=base();r[22]=0;c.push_back({"zero captured extent refuses",r,false});
    return c;
}
inline bool compare(const Case& test,const cc::Result& actual,std::string& error)
{
    const auto& input=test.input;const auto bound=cc::maximumVisits(input[1]);
    const bool admitted=input[0]==1&&bound&&input[2]>=bound&&!input[3]&&input[8]<=16777216u&&input[9]<=16777216u;
    const auto fail=[&](const char* reason){error=std::string(test.label)+": "+reason;return false;};
    if(actual[1]!=(admitted?bound:0)||actual[2]>actual[1])return fail("work envelope differs");
    if(!test.supported){if(actual[0])return fail("unsupported model/domain accepted");
        for(std::size_t k=3;k<actual.size();k++)if(actual[k])return fail("refusal leaked partial coefficients");
        return true;}
    if(actual[0]!=1||actual[3]!=input[1])return fail("supported arithmetic refused");
    for(unsigned k=0;k<cc::kRequestWords;k++)if(actual[4+k]!=input[k])return fail("source/endpoint provenance changed");
    const auto model=oracle(input);if(!model.valid)return fail("independent model invalid");
    for(unsigned form=0;form<17;form++){
        const auto at=61+form*28;bool odd=false,nonzero=false;
        const auto exponent=actual[at];const auto signedExponent=exponent<=0x7fffffffu?static_cast<std::int64_t>(exponent):static_cast<std::int64_t>(exponent)-4294967296ll;
        if(signedExponent<cc::kMinimumExponent||signedExponent>cc::kMaximumExponent)return fail("form exponent outside domain");
        for(unsigned component=0;component<3;component++){
            const auto sign=actual[at+1+9*component];const auto* words=actual.data()+at+2+9*component;
            if(sign>2)return fail("invalid coefficient sign");
            bool any=false;for(unsigned k=0;k<8;k++)any|=words[k]!=0;
            if((sign!=0)!=any)return fail("noncanonical coefficient zero");
            if(input[1]==30&&(words[0]>>30))return fail("small coefficient exceeds 30 bits");
            if(input[1]==30)for(unsigned k=1;k<8;k++)if(words[k])return fail("small coefficient exceeds one word");
            odd|=(words[0]&1u)!=0;nonzero|=any;
            if(!equal(readScalar(exponent,sign,words),model.forms[form][component]))return fail("independent rational form differs");}
        if(nonzero&&!odd)return fail("common positive dyadic scale is not canonical");
        if(!nonzero&&exponent)return fail("zero form exponent is not canonical");
    }
    for(unsigned tri=0;tri<2;tri++){const auto at=537+10*tri;
        const auto exponent=actual[at];const auto e=exponent<=0x7fffffffu?static_cast<std::int64_t>(exponent):static_cast<std::int64_t>(exponent)-4294967296ll;
        if(e<cc::kMinimumExponent||e>cc::kMaximumExponent||actual[at+1]<1||actual[at+1]>2||(actual[at+2]&1u)==0)
            return fail("noncanonical H");
        if(input[1]==30){if(actual[at+2]>>30)return fail("H exceeds small precision");
            for(unsigned k=1;k<8;k++)if(actual[at+2+k])return fail("H exceeds small word");}
        if(!equal(readScalar(actual[at],actual[at+1],actual.data()+at+2),model.heights[tri]))return fail("independent H differs");}
    return true;
}
}
