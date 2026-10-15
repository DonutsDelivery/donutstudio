#pragma once
#include "fixture_native_exact_line_predicates.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <limits>

namespace arbitgpu::lambertian
{
// Private mathematics over supplied exitance. No metre, native parent, completed
// command, resource, material, base draw, radiance receipt or Color authority.
constexpr std::uint32_t kVersion=1,kP32Bits=0x40490fdbu,kP32Significand=13176795u;
constexpr unsigned kBits=640,kLimbs=20,kRequestWords=22,kResultWords=246;
using Request=std::array<std::uint32_t,kRequestWords>;
using Result=std::array<std::uint32_t,kResultWords>;
// Request: version/P32/precision128/capacity640/exponent(two's complement),
// four little-endian significand words, reserved0, six low/high budget pairs.
// The odd significand is canonical. Zero requires exponent0. Exponent -253..17.
// Result: status/floor quotient/certified bits/binade, exact request echo,
// eleven 20-word magnitudes: scaled N/D/remainder, pi D/L/U, interval numerator,
// lower denominator/U, upper denominator/L, error numerator/denominator.
// Failed admitted evaluation zeros mathematics, retaining the echo. Insufficient
// admission leaves output untouched and returns false before parsing or clearing.
struct Ceilings {
    std::uint64_t limbVisits,wordMoves,scalarOperations,parserOperations,comparisons,scratchBytes;
};
// Census BEFORE execution, including failed calls. D construction <=30 products;
// 13 term denominators <=13*19 products; term weighting/sums and all conversion,
// interval, reciprocal selection and error work bring products below350. There
// are13 exact pi divisions and1 rounding division. Each has640 iterations with
// one shift, magnitude comparison and optional signed sum. Additional shifts
// <=400 include143 binade comparisons, denominators and all dyadic alignment.
// <=100 sums outside division; <=300 comparisons outside division. All helpers
// use the existing35-word representation, checked against20 active words.
constexpr std::uint64_t kProducts=350,kDivisions=14,kShifts=400+kDivisions*kBits;
constexpr std::uint64_t kSums=100+kDivisions*kBits,kComparisons=300+kDivisions*kBits;
constexpr std::uint64_t kShiftVisits=35+kBits+2*35;
constexpr std::uint64_t kNativeVisits=kProducts*exactline::multiplyVisits(20,20)
    +kShifts*kShiftVisits+kSums*175+kComparisons*35+kDivisions*kBits*35+30000;
// Comparison admission uses the full scalar ceiling, covering both limb
// comparisons and every scalar/control/parser/copy-loop comparison.
// Copies charge value/reference copy-in/out, even when optimized away. Product
// <=155+240+400*(8+2*35*82+2*35*6)+8*28575 words, including nested ELWork
// copies. Shift <=8*745+512; sum <=8*175+1000; compare <=8*35+512.
// 100000 fixed moves cover result/echo, parser, Small calls, local assignments,
// division bit writes, integer sign changes, failure/zero paths and bit-cast.
constexpr std::uint64_t kNativeMoves=100000+kProducts*(155+240+400*(8+70*82+70*6)+8*28575)
    +kShifts*(8*kShiftVisits+512)+kSums*(8*175+1000)+kComparisons*(8*35+512)
    +kDivisions*kBits*(512+8*35);
constexpr std::uint64_t kNativeParser=4096;
constexpr std::uint64_t kNativeScalars=10000000+64*kNativeVisits+8*kNativeMoves;
// <=96 simultaneously live ELInt values/copies at144 bytes, 8 request/result
// copies and4096 scalar bytes. This includes caller entry argument copies and
// rational/division nesting, not compiled spills, source preparation or owners.
constexpr Ceilings kNativeCeilings{kNativeVisits,kNativeMoves,kNativeScalars,kNativeParser,
    kNativeScalars,96*144+8*(kRequestWords+kResultWords)*4+4096};
static_assert(kNativeVisits<0xffffffffull,"ELWork counter cannot wrap");
static_assert(kNativeVisits==19227650&&kNativeMoves==1040520410&&kNativeScalars==9564732880ull
    &&kNativeCeilings.scratchBytes==26496,"native attempted schedule census");
// Independent host verifier uses20-word integers and uint64 products. Its
// schedule allows600 products,14 divisions,12000 shifts/sums/comparisons and
// 80000 fixed word scans/copies. Every host operation walks fixed20 limbs.
// multiply<=20 clear+400 products+400*20 carry visits; divide<=640*(60+20+20).
// <=128 bit-length scans of640 positions cover independent dyadic alignment.
// Four common-power cancellations reserve640 rounds of40 word visits each.
constexpr std::uint64_t kHostVisits=600*(20+400+8000)+14*640*100+12000*100+128*kBits+4*kBits*40+80000;
constexpr std::uint64_t kHostMoves=32*kHostVisits+100000;
constexpr Ceilings kHostCeilings{kHostVisits,kHostMoves,128*kHostVisits+8*kHostMoves+1000000,
    8192,128*kHostVisits+8*kHostMoves+1000000,96*80+8*(kRequestWords+kResultWords)*4+4096};
static_assert(kHostVisits==7412320&&kHostMoves==237294240&&kHostCeilings.scalarOperations==2848130880ull
    &&kHostCeilings.scratchBytes==20352,"independent host attempted schedule census");
inline bool admitted(const Ceilings& a,const Ceilings& b)
{return a.limbVisits>=b.limbVisits&&a.wordMoves>=b.wordMoves&&a.scalarOperations>=b.scalarOperations
    &&a.parserOperations>=b.parserOperations&&a.comparisons>=b.comparisons&&a.scratchBytes>=b.scratchBytes;}
inline void putBudgets(Request& q,const Ceilings& c)
{
    const std::uint64_t words[]{c.limbVisits,c.wordMoves,c.scalarOperations,c.parserOperations,c.comparisons,c.scratchBytes};
    for(unsigned i=0;i<6;i++){q[10+2*i]=std::uint32_t(words[i]);q[11+2*i]=std::uint32_t(words[i]>>32);}
}
inline bool nativeAdmitted(const Request& q)
{
    const Ceilings a{std::uint64_t(q[10])|(std::uint64_t(q[11])<<32),std::uint64_t(q[12])|(std::uint64_t(q[13])<<32),
        std::uint64_t(q[14])|(std::uint64_t(q[15])<<32),std::uint64_t(q[16])|(std::uint64_t(q[17])<<32),
        std::uint64_t(q[18])|(std::uint64_t(q[19])<<32),std::uint64_t(q[20])|(std::uint64_t(q[21])<<32)};
    return admitted(a,kNativeCeilings);
}
// Pi uses D=9*5*7*11*13*17*5^17*239^7. All odd factors1..17
// divide9*5*7*11*13*17. Every one of the13 term denominators divides D.
// D<2^118; pi numerators<2^120. Each term denominator<2^61.
// 16*a-4*b is lower, with8 alternating a terms and3 alternating b terms.
// Upper adds16/(17*5^17)+4/(7*239^7). Alternating remainders and
// Machin's identity16*atan(1/5)-4*atan(1/239)=pi establish the enclosure.
// Native scaled conversion N/D widths <=316/277. Binade comparisons <=295 bits.
// Physical M*D <=263 bits,65504*piL*Mden <=389 bits. Interval widths
// <=263/373. Reciprocal error numerator/denominator <=143/144 bits;
// its comparison <=287 bits. Choose that maximum BEFORE multiplying M.
// M*error <=288/397 bits. Radius <=2^-9 and >=2^-150. Addition uses
// <=548-bit numerator/denominator intermediates, within checked640 bits.
// These loose widths cover cancellation, exponent extremes and neighboring
// subnormal at the minimum NORMAL output. Zero also uses radius2^-150.
// Every helper still checks capacity.
inline const char* arithmeticSource()
{
    return R"LAMBERTIAN(
// Requires the unchanged EL arithmetic body and LR_ADMIT(request).
// LR_REQUEST/LR_RESULT are thread-array references in Metal. No floating-point division.
void lrSmall(uint n,EL_REF(ELInt,r),EL_REF(ELWork,w)) {elZero(r,w);r.limb[0]=n;r.sign=n==0u?0u:1u;}
void lrShift(EL_VAL(ELInt,a),uint s,EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    elZero(r,w);r.sign=a.sign;
    for(uint bit=0u;bit<640u;bit++){elStep(w);if((a.limb[bit/32u]&(1u<<(bit%32u)))!=0u){
        if(s>=640u||bit>=640u-s){w.ok=false;continue;}
        uint at=bit+s;r.limb[at/32u]|=1u<<(at%32u);}}
    elCapacity(a,20u,w);elCapacity(r,20u,w);
}
void lrMul(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_REF(ELInt,r),EL_REF(ELWork,w))
{elMultiply(a,b,20u,20u,20u,r,w);}
void lrScale(EL_VAL(ELInt,a),uint n,EL_REF(ELInt,r),EL_REF(ELWork,w))
{ELInt b;lrSmall(n,b,w);lrMul(a,b,r,w);}
void lrSub(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_REF(ELInt,r),EL_REF(ELWork,w))
{if(b.sign!=0u)b.sign=2u;elSum(a,b,r,w);if(r.sign==2u)w.ok=false;}
void lrDivide(EL_VAL(ELInt,n),EL_VAL(ELInt,d),EL_REF(ELInt,q),EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    elZero(q,w);elZero(r,w);if(d.sign!=1u){w.ok=false;return;}
    for(int bit=639;bit>=0;bit--){ELInt shifted;lrShift(r,1u,shifted,w);r=shifted;
        r.limb[0]|=(n.limb[bit/32]>>uint(bit%32))&1u;r.sign=1u;elNormalize(r,w);
        if(elMagnitude(r,d,w)>=0){ELInt difference;lrSub(r,d,difference,w);r=difference;
            q.limb[bit/32]|=1u<<uint(bit%32);}}
    q.sign=1u;elNormalize(q,w);
}
void lrPi(EL_REF(ELInt,d),EL_REF(ELInt,l),EL_REF(ELInt,u),EL_REF(ELWork,w)) {
    lrSmall(1u,d,w);uint factors[6]={9u,5u,7u,11u,13u,17u};
    for(int k=0;k<30;k++){if(k>=6&&k<23){ELInt t;lrScale(d,5u,t,w);d=t;}
        else if(k>=23){ELInt t;lrScale(d,239u,t,w);d=t;}
        else{ELInt t;lrScale(d,factors[k],t,w);d=t;}}
    // Signed partial sums are used only here. piL stays positive after a's first term.
    elZero(l,w);ELInt extra;elZero(extra,w);
    for(int k=0;k<13;k++){
        uint base=k<8||k==11?5u:239u;
        uint index=k<8?uint(k):k<11?uint(k-8):k==11?8u:3u;
        ELInt denom;lrSmall(1u,denom,w);
        for(uint j=0u;j<17u;j++)if(j<2u*index+1u){ELInt t;lrScale(denom,base,t,w);denom=t;}
        ELInt t;lrScale(denom,2u*index+1u,t,w);denom=t;
        ELInt term,rem;lrDivide(d,denom,term,rem,w);if(rem.sign!=0u)w.ok=false;
        lrScale(term,k<8||k==11?16u:4u,t,w);term=t;
        if(k<11){if(k<8?(index&1u)!=0u:(index&1u)==0u)term.sign=2u;
            elSum(l,term,t,w);l=t;}
        else{elSum(extra,term,t,w);extra=t;}
    }
    elSum(l,extra,u,w);if(l.sign!=1u||elMagnitude(l,u,w)>=0)w.ok=false;
}
bool lrEvaluate(LR_REQUEST(request),LR_RESULT(result)) {
    if(!LR_ADMIT(request))return false;
    for(int k=0;k<246;k++)result[k]=0u;
    for(int k=0;k<22;k++)result[4+k]=request[k];
    if(request[0]!=1u||request[1]!=0x40490fdbu||request[2]!=128u||request[3]!=640u||request[9]!=0u)return true;
    uint ew=request[4];if(ew>17u&&ew<4294967043u)return true;
    int exponent=ew<=17u?int(ew):-int((~ew)+1u);
    bool zero=(request[5]|request[6]|request[7]|request[8])==0u;
    if(zero?exponent!=0:(request[5]&1u)==0u)return true;
    ELWork w;w.used=0u;w.limit=LR_VISITS;w.capacity=20u;w.ok=true;
    ELInt piD,piL,piU;lrPi(piD,piL,piU,w);if(!w.ok)return true;
    ELInt m,md;elZero(m,w);for(int k=0;k<4;k++)m.limb[k]=request[5+k];m.sign=zero?0u:1u;lrSmall(1u,md,w);
    ELInt t;if(exponent>=0){lrShift(m,uint(exponent),t,w);m=t;}else{lrShift(md,uint(-exponent),t,w);md=t;}
    ELInt physical,limit;lrMul(m,piD,physical,w);lrMul(md,piL,t,w);lrScale(t,65504u,limit,w);
    if(!w.ok||elMagnitude(physical,limit,w)>0)return true;
    ELInt n,d;lrShift(m,22u,n,w);lrScale(md,13176795u,d,w);
    int binade=-126;
    for(int e=-125;e<=16;e++){ELInt a,b;if(e>=0){a=n;lrShift(d,uint(e),b,w);}else{lrShift(n,uint(-e),a,w);b=d;}
        if(elMagnitude(a,b,w)>=0)binade=e;}
    ELInt scaledN,scaledD;if(binade>=23){scaledN=n;lrShift(d,uint(binade-23),scaledD,w);}
    else{lrShift(n,uint(23-binade),scaledN,w);scaledD=d;}
    ELInt quotient,rem;lrDivide(scaledN,scaledD,quotient,rem,w);
    for(int k=1;k<35;k++)if(quotient.limb[k]!=0u)w.ok=false;
    uint floorQ=quotient.limb[0];lrShift(rem,1u,t,w);int midpoint=elMagnitude(t,scaledD,w);
    uint rounded=floorQ+(midpoint>0||(midpoint==0&&(floorQ&1u)!=0u)?1u:0u);
    if(rounded==16777216u){rounded=8388608u;binade++;}
    if(!zero&&(rounded<8388608u||rounded>=16777216u||binade<-126||binade>127))return true;
    uint bits=zero?0u:(uint(binade+127)<<23u)|(rounded-8388608u);
    // Reciprocal errors |2^22*piN-Psig*piD|/(Psig*piN).
    ELInt en[2],ed[2];
    for(int k=0;k<2;k++){ELInt pn=k==0?piL:piU,a,b;lrShift(pn,22u,a,w);lrScale(piD,13176795u,b,w);
        if(elMagnitude(a,b,w)>=0)lrSub(a,b,en[k],w);else lrSub(b,a,en[k],w);
        lrScale(pn,13176795u,ed[k],w);}
    ELInt a,b;lrMul(en[0],ed[1],a,w);lrMul(en[1],ed[0],b,w);int selected=elMagnitude(a,b,w)>=0?0:1;
    ELInt errorN,errorD;lrMul(m,en[selected],errorN,w);lrMul(md,ed[selected],errorD,w);
    // Largest adjacent gap/2. Normal powers of two have a smaller lower gap.
    // At the smallest normal, the subnormal neighbor still has gap2^-149.
    int radiusPower=zero?-150:binade-24;ELInt radiusN,radiusD;lrSmall(1u,radiusN,w);lrSmall(1u,radiusD,w);
    if(radiusPower>=0){lrShift(radiusN,uint(radiusPower),t,w);radiusN=t;}
    else{lrShift(radiusD,uint(-radiusPower),t,w);radiusD=t;}
    lrMul(errorN,radiusD,a,w);lrMul(radiusN,errorD,b,w);elSum(a,b,t,w);errorN=t;
    lrMul(errorD,radiusD,t,w);errorD=t;
    ELInt lowerD,upperD;lrMul(md,piU,lowerD,w);lrMul(md,piL,upperD,w);
    if(!w.ok)return true;
    result[0]=1u;result[1]=floorQ;result[2]=bits;result[3]=uint(binade);
    for(int k=0;k<20;k++){
        result[26+k]=scaledN.limb[k];result[46+k]=scaledD.limb[k];result[66+k]=rem.limb[k];
        result[86+k]=piD.limb[k];result[106+k]=piL.limb[k];result[126+k]=piU.limb[k];
        result[146+k]=physical.limb[k];result[166+k]=lowerD.limb[k];result[186+k]=upperD.limb[k];
        result[206+k]=errorN.limb[k];result[226+k]=errorD.limb[k];}
    return true;
}
)LAMBERTIAN";
}
// Source assembly/caller retention and compiled code are outside these arithmetic
// ceilings. The future native owner must admit them separately before assembly.
inline std::string metalArithmeticSource()
{
    auto source=exactline::languageSource(true);
    source+="\n#define LR_REQUEST(N) thread const uint (&N)[22]\n#define LR_RESULT(N) thread uint (&N)[246]\n";
    source+="#define LR_VISITS "+std::to_string(kNativeVisits)+"u\n";
    source+="bool lrAdmit(LR_REQUEST(q)){\n";
    const std::uint64_t values[]{kNativeCeilings.limbVisits,kNativeCeilings.wordMoves,kNativeCeilings.scalarOperations,
        kNativeCeilings.parserOperations,kNativeCeilings.comparisons,kNativeCeilings.scratchBytes};
    for(unsigned i=0;i<6;i++)source+="if(q["+std::to_string(11+2*i)+"]<"+std::to_string(values[i]>>32)+
        "u||(q["+std::to_string(11+2*i)+"]=="+std::to_string(values[i]>>32)+"u&&q["+
        std::to_string(10+2*i)+"]<"+std::to_string(std::uint32_t(values[i]))+"u))return false;\n";
    source+="return true;}\n#define LR_ADMIT(Q) lrAdmit(Q)\n";
    return source+arithmeticSource();
}
namespace host
{
// Independent fixed-storage implementation. It does not invoke native helpers,
// native division, native binade selection or floating-point division. uint64
// word products/carries replace the native split16 helpers. No heap integers.
struct Integer {std::array<std::uint32_t,kLimbs> word{};};
struct Work {bool ok=true;};
inline Integer small(std::uint32_t n){Integer a;a.word[0]=n;return a;}
inline bool zero(const Integer& a){std::uint32_t v=0;for(auto n:a.word)v|=n;return v==0;}
inline int compare(const Integer& a,const Integer& b)
{int r=0;for(unsigned k=kLimbs;k-->0;)if(r==0&&a.word[k]!=b.word[k])r=a.word[k]>b.word[k]?1:-1;return r;}
inline Integer add(const Integer& a,const Integer& b,Work& w)
{
    Integer r;std::uint64_t carry=0;
    for(unsigned k=0;k<kLimbs;k++){const auto s=std::uint64_t(a.word[k])+b.word[k]+carry;r.word[k]=std::uint32_t(s);carry=s>>32;}
    if(carry)w.ok=false;
    return r;
}
inline Integer subtract(const Integer& a,const Integer& b,Work& w)
{
    Integer r;std::uint64_t borrow=0;
    for(unsigned k=0;k<kLimbs;k++){const auto x=std::uint64_t(b.word[k])+borrow;
        r.word[k]=std::uint32_t(std::uint64_t(a.word[k])-x);borrow=std::uint64_t(a.word[k])<x?1:0;}
    if(borrow)w.ok=false;
    return r;
}
inline Integer shift(const Integer& a,unsigned n,Work& w)
{
    Integer r;
    for(unsigned k=0;k<kLimbs;k++){
        const unsigned at=k+n/32,offset=n%32;
        if(at>=kLimbs){if(a.word[k])w.ok=false;continue;}
        r.word[at]|=a.word[k]<<offset;
        if(offset){const auto high=a.word[k]>>(32-offset);if(at+1>=kLimbs){if(high)w.ok=false;}
            else r.word[at+1]|=high;}
    }
    return r;
}
inline Integer multiply(const Integer& a,const Integer& b,Work& w)
{
    Integer r;
    for(unsigned i=0;i<kLimbs;i++)for(unsigned j=0;j<kLimbs;j++){
        const auto p=std::uint64_t(a.word[i])*b.word[j];
        std::uint64_t carry=p;
        for(unsigned t=0;t<kLimbs;t++){
            const unsigned at=i+j+t;
            if(at>=kLimbs){if(carry)w.ok=false;break;}
            // low32(carry)+word <=2^33-2; separate high avoids uint64 overflow.
            const auto s=std::uint64_t(r.word[at])+std::uint32_t(carry);
            r.word[at]=std::uint32_t(s);carry=(carry>>32)+(s>>32);
        }
        if(carry)w.ok=false;
    }
    return r;
}
inline Integer scale(const Integer& a,std::uint32_t n,Work& w){return multiply(a,small(n),w);}
struct Division {Integer quotient,remainder;};
inline Division divide(const Integer& n,const Integer& d,Work& w)
{
    Division r;if(zero(d)){w.ok=false;return r;}
    for(unsigned k=kBits;k-->0;){r.remainder=shift(r.remainder,1,w);
        r.remainder.word[0]|=(n.word[k/32]>>(k%32))&1u;
        if(compare(r.remainder,d)>=0){r.remainder=subtract(r.remainder,d,w);r.quotient.word[k/32]|=1u<<(k%32);}}
    return r;
}
struct Pi {Integer denominator,lower,upper;};
inline Pi pi(Work& w)
{
    Pi p;p.denominator=small(1);
    for(auto f:std::array<std::uint32_t,6>{{9,5,7,11,13,17}})p.denominator=scale(p.denominator,f,w);
    for(unsigned i=0;i<17;i++)p.denominator=scale(p.denominator,5,w);
    for(unsigned i=0;i<7;i++)p.denominator=scale(p.denominator,239,w);
    // Construct positive and negative sums independently, then subtract once.
    Integer positive,negative,tail;
    for(unsigned family=0;family<2;family++)for(unsigned i=0;i<(family?4u:9u);i++){
        Integer d=small(2*i+1);
        for(unsigned j=0;j<2*i+1;j++)d=scale(d,family?239:5,w);
        const auto fraction=divide(p.denominator,d,w);if(!zero(fraction.remainder))w.ok=false;
        const auto term=scale(fraction.quotient,family?4:16,w);
        if(i==(family?3u:8u))tail=add(tail,term,w);
        else if(family?(i%2==0):(i%2!=0))negative=add(negative,term,w);
        else positive=add(positive,term,w);
    }
    p.lower=subtract(positive,negative,w);p.upper=add(p.lower,tail,w);
    if(zero(p.lower)||compare(p.lower,p.upper)>=0)w.ok=false;
    return p;
}
struct Rational {Integer numerator,denominator=small(1);};
inline Rational number(std::uint32_t n){return {small(n),small(1)};}
inline Rational sum(const Rational& a,const Rational& b,Work& w)
{return {add(multiply(a.numerator,b.denominator,w),multiply(b.numerator,a.denominator,w),w),multiply(a.denominator,b.denominator,w)};}
inline int compare(const Rational& a,const Rational& b,Work& w)
{return compare(multiply(a.numerator,b.denominator,w),multiply(b.numerator,a.denominator,w));}
inline Rational floatValue(std::uint32_t bits,Work& w)
{
    const auto e=(bits>>23)&255u;Rational r=number(e?8388608u|(bits&8388607u):bits&8388607u);
    const int power=e?int(e)-150:-149;
    if(power<0)r.denominator=shift(r.denominator,unsigned(-power),w);
    else r.numerator=shift(r.numerator,unsigned(power),w);
    return r;
}
inline Rational cancelTwos(Rational r)
{
    // Fixed640-round bound includes the all-zero numerator. Denominator is
    // positive, so it never shifts to zero.
    for(unsigned round=0;round<kBits;round++)if((r.numerator.word[0]&1u)==0&&(r.denominator.word[0]&1u)==0){
        for(unsigned k=0;k<kLimbs;k++){
            r.numerator.word[k]=(r.numerator.word[k]>>1)|(k+1<kLimbs?r.numerator.word[k+1]<<31:0u);
            r.denominator.word[k]=(r.denominator.word[k]>>1)|(k+1<kLimbs?r.denominator.word[k+1]<<31:0u);
        }
    }
    return r;
}
inline unsigned bitLength(const Integer& a)
{
    unsigned length=0;
    for(unsigned k=0;k<kBits;k++)if(a.word[k/32]&(1u<<(k%32)))length=k+1;
    return length;
}
inline Rational midpoint(std::uint32_t lo,std::uint32_t hi,Work& w)
{
    // Align dyadics at the larger denominator instead of multiplying powers of
    // two. This keeps minimum-normal midpoint comparisons within640 bits.
    auto a=floatValue(lo,w),b=floatValue(hi,w);
    const bool aSmaller=compare(a.denominator,b.denominator)<0;
    const auto& larger=aSmaller?b.denominator:a.denominator;
    const auto& smaller=aSmaller?a.denominator:b.denominator;
    const unsigned alignment=bitLength(larger)-bitLength(smaller);
    const auto n=add(aSmaller?shift(a.numerator,alignment,w):a.numerator,
        aSmaller?b.numerator:shift(b.numerator,alignment,w),w);
    return {n,shift(larger,1,w)};
}
inline std::uint32_t rounded(const Rational& value,Work& w)
{
    // Independent monotone search of actual binary32 encodings, followed by an
    // exact adjacent midpoint/parity decision. No native quotient logic reused.
    std::uint32_t lo=0,hi=0x47800000u;
    for(unsigned k=0;k<31;k++)if(lo+1<hi){const auto mid=lo+(hi-lo)/2;
        if(compare(value,floatValue(mid,w),w)>=0)lo=mid;else hi=mid;}
    const int c=compare(value,midpoint(lo,lo+1,w),w);
    return lo+(c>0||(c==0&&(lo&1u))?1u:0u);
}
inline Integer read(const Result& r,unsigned index)
{Integer a;for(unsigned k=0;k<kLimbs;k++)a.word[k]=r[26+20*index+k];return a;}
inline bool same(const Integer& a,const Integer& b){return compare(a,b)==0;}
inline bool canonical(const Request& q)
{
    if(q[0]!=kVersion||q[1]!=kP32Bits||q[2]!=128||q[3]!=640||q[9])return false;
    if(q[4]>17u&&q[4]<0xffffff03u)return false;
    const bool z=(q[5]|q[6]|q[7]|q[8])==0;
    return z?q[4]==0:(q[5]&1u)!=0;
}
inline Rational exitance(const Request& q,Work& w)
{
    Rational m;for(unsigned k=0;k<4;k++)m.numerator.word[k]=q[5+k];
    const int e=q[4]<=17?int(q[4]):-int((~q[4])+1u);
    if(e>=0)m.numerator=shift(m.numerator,unsigned(e),w);
    else m.denominator=shift(m.denominator,unsigned(-e),w);
    return m;
}
} // namespace host

enum class Inspection {NotAdmitted,Invalid,Refused,Certified};
inline Inspection inspect(const Request& q,const Result& result,const Ceilings& allowance)
{
    // Admission uses scalar comparisons only, before record copying or parsing.
    if(!admitted(allowance,kHostCeilings)||!nativeAdmitted(q))return Inspection::NotAdmitted;
    for(unsigned k=0;k<kRequestWords;k++)if(result[4+k]!=q[k])return Inspection::Invalid;
    if(result[0]>1)return Inspection::Invalid;
    auto refusal=[&](){for(unsigned k=1;k<4;k++)if(result[k])return Inspection::Invalid;
        for(unsigned k=26;k<kResultWords;k++)if(result[k])return Inspection::Invalid;
        return result[0]==0?Inspection::Refused:Inspection::Invalid;};
    if(!host::canonical(q))return refusal();
    host::Work w;const auto pi=host::pi(w);const auto m=host::exitance(q,w);
    const auto physical=host::multiply(m.numerator,pi.denominator,w);
    const auto limit=host::scale(host::multiply(m.denominator,pi.lower,w),65504,w);
    if(!w.ok)return Inspection::Invalid;
    if(host::compare(physical,limit)>0)return refusal();
    const host::Rational ratio{host::shift(m.numerator,22,w),host::scale(m.denominator,kP32Significand,w)};
    const auto bits=host::rounded(ratio,w);
    if(!w.ok)return Inspection::Invalid;
    const bool z=host::zero(m.numerator);
    if(!z&&(bits<0x00800000u||bits>=0x7f800000u))return refusal();
    if(result[0]!=1||result[2]!=bits)return Inspection::Invalid;
    const int binade=z?-126:int((bits>>23)&255u)-127;
    if(result[3]!=std::uint32_t(binade))return Inspection::Invalid;
    if(!host::same(host::read(result,3),pi.denominator)||!host::same(host::read(result,4),pi.lower)
        ||!host::same(host::read(result,5),pi.upper))return Inspection::Invalid;
    // Recover the scale BEFORE a rounding carry. It differs only if the result
    // is a binade boundary and the exact value is below that boundary.
    int floorBinade=binade;
    if(!z&&binade>-126&&host::compare(ratio,host::floatValue(bits,w),w)<0&&(bits&8388607u)==0)floorBinade--;
    host::Integer n=ratio.numerator,d=ratio.denominator;
    if(floorBinade<23)n=host::shift(n,unsigned(23-floorBinade),w);
    else d=host::shift(d,unsigned(floorBinade-23),w);
    const auto rem=host::read(result,2),quotient=host::small(result[1]);
    if(!host::same(n,host::read(result,0))||!host::same(d,host::read(result,1))
        ||host::compare(rem,d)>=0||!host::same(n,host::add(host::multiply(quotient,d,w),rem,w)))return Inspection::Invalid;
    const int halfway=host::compare(host::shift(rem,1,w),d);
    const auto significand=result[1]+(halfway>0||(halfway==0&&(result[1]&1u))?1u:0u);
    if(!z&&significand!=(floorBinade==binade?(bits&8388607u)|8388608u:16777216u))return Inspection::Invalid;
    // Explicit adjacent midpoint certification also covers the min-normal
    // subnormal neighbor. Ties belong to the even encoding on both sides.
    if(!z){const auto lower=host::compare(ratio,host::midpoint(bits-1,bits,w),w);
        const auto upper=host::compare(ratio,host::midpoint(bits,bits+1,w),w);
        if(lower<0||upper>0||((lower==0||upper==0)&&(bits&1u)))return Inspection::Invalid;}
    if(!host::same(host::read(result,6),physical)
        ||!host::same(host::read(result,7),host::multiply(m.denominator,pi.upper,w))
        ||!host::same(host::read(result,8),host::multiply(m.denominator,pi.lower,w)))return Inspection::Invalid;
    std::array<host::Rational,2> errors;
    const std::array<host::Integer,2> pn{{pi.lower,pi.upper}};
    for(unsigned k=0;k<2;k++){
        const auto a=host::shift(pn[k],22,w),b=host::scale(pi.denominator,kP32Significand,w);
        errors[k]={host::compare(a,b)>=0?host::subtract(a,b,w):host::subtract(b,a,w),host::scale(pn[k],kP32Significand,w)};
    }
    const auto& maximum=host::compare(errors[0],errors[1],w)>=0?errors[0]:errors[1];
    const host::Rational piError{host::multiply(m.numerator,maximum.numerator,w),host::multiply(m.denominator,maximum.denominator,w)};
    // Radius derives from adjacent floats independently of native binade logic.
    host::Rational radius{host::small(1),host::shift(host::small(1),150,w)};
    if(!z){const auto center=host::floatValue(bits,w),before=host::floatValue(bits-1,w),after=host::floatValue(bits+1,w);
        auto gap=[&](const host::Rational& high,const host::Rational& low){
            // Both are dyadics. Align by bit length; do not multiply denominators.
            const bool highD=host::compare(high.denominator,low.denominator)>=0;
            const auto& dmax=highD?high.denominator:low.denominator;
            const auto& dmin=highD?low.denominator:high.denominator;
            const unsigned factor=host::bitLength(dmax)-host::bitLength(dmin);
            return host::Rational{host::subtract(highD?high.numerator:host::shift(high.numerator,factor,w),
                highD?host::shift(low.numerator,factor,w):low.numerator,w),host::shift(dmax,1,w)};};
        const auto below=gap(center,before),above=gap(after,center);
        radius=host::compare(below,above,w)>=0?below:above;
    }
    // Native uses unreduced2^(-binade+24), independently checked against radius.
    host::Rational nativeRadius=host::number(1);
    nativeRadius.denominator=host::shift(host::small(1),unsigned(z?150:24-binade),w);
    if(host::compare(radius,nativeRadius,w)!=0)return Inspection::Invalid;
    const auto error=host::sum(piError,nativeRadius,w);
    if(!host::same(host::read(result,9),error.numerator)||!host::same(host::read(result,10),error.denominator))return Inspection::Invalid;
    // Check both exact physical endpoints against the error, after canceling
    // shared powers of two. Direct unreduced cross products can exceed640 bits.
    // Tight constant proof:765765<3*2^18,5^17<3*2^38,
    // 239^7<240^7=15^7*2^28<3*2^54. Thus D<27*2^110<2^115,
    // piN<2^117, reciprocal-error N/D<2^139/2^141. Normal admission
    // implies Md<=2^252. Aligning M and v gives <=128-bit integers because
    // RN-even normal v<M, P32>3, and an aligned float coefficient has24 bits.
    // Endpoint difference N/D <=246/369 bits. The canceled error <=269/393
    // bits. Both cross products are <=639 bits, including signed difference.
    const auto reducedError=host::cancelTwos(error);
    const auto center=host::floatValue(bits,w);
    const bool mD=host::compare(m.denominator,center.denominator)>=0;
    const auto& common=mD?m.denominator:center.denominator;
    const unsigned distance=host::bitLength(common)-host::bitLength(mD?center.denominator:m.denominator);
    const auto mAligned=mD?m.numerator:host::shift(m.numerator,distance,w);
    const auto vAligned=mD?host::shift(center.numerator,distance,w):center.numerator;
    for(const auto& endpoint:pn){
        const auto a=host::multiply(vAligned,endpoint,w),b=host::multiply(mAligned,pi.denominator,w);
        const auto absolute=host::compare(a,b)>=0?host::subtract(a,b,w):host::subtract(b,a,w);
        const auto difference=host::cancelTwos(host::Rational{absolute,host::multiply(common,endpoint,w)});
        if(host::compare(difference,reducedError,w)>0)return Inspection::Invalid;
    }
    return w.ok?Inspection::Certified:Inspection::Invalid;
}
// The only float conversion API accepts a fully inspected record. Bit-casting
// does not attest actual native execution. Caller must bind readback separately.
inline bool certifiedValue(const Request& q,const Result& r,const Ceilings& allowance,float& value)
{
    if(inspect(q,r,allowance)!=Inspection::Certified)return false;
    static_assert(sizeof(float)==sizeof(std::uint32_t)&&std::numeric_limits<float>::is_iec559
        &&std::numeric_limits<float>::digits==24&&std::numeric_limits<float>::max_exponent==128,"binary32 storage");
    const auto bits=r[2];std::memcpy(&value,&bits,sizeof(bits));return true;
}
} // namespace arbitgpu::lambertian
