#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace videorender::rotatingexact
{
// Only the active words participate. Operations write their active destination
// themselves; default construction does not clear or copy a full-width value.
struct Integer {
    std::array<std::uint32_t,87> word;
    std::uint32_t negative;
};
struct Workspace { std::array<std::uint32_t,174> product; };
static_assert(sizeof(Integer)==352 && alignof(Integer)==alignof(std::uint32_t));
static_assert(sizeof(Workspace)==696);
static_assert(std::is_trivial_v<Integer> && std::is_standard_layout_v<Integer>);
static_assert(sizeof(float)==4 && sizeof(double)==8
    && std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits==24
    && std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53);

// Checked source units count active limb visits, multiply cells and streamed
// bit transitions, rather than machine instructions or elapsed time. Callers
// must admit the complete price before invoking a primitive. Failed calls keep
// that attempted price. These primitives confer no replay authority.
struct Cost {
    static constexpr bool supported(unsigned n) { return n>=1 && n<=87; }
    static constexpr auto unavailable=std::numeric_limits<std::uint64_t>::max();
    static constexpr std::uint64_t decode(unsigned n) { return supported(n) ? n+12:unavailable; }
    static constexpr std::uint64_t add(unsigned n) { return supported(n) ? 3*n+8:unavailable; }
    static constexpr std::uint64_t compare(unsigned n) { return supported(n) ? n+4:unavailable; }
    static constexpr std::uint64_t shift(unsigned n) { return supported(n) ? n+6:unavailable; }
    static constexpr std::uint64_t multiply(unsigned n) { return supported(n) ? std::uint64_t(n)*n+5*n+12:unavailable; }
    static constexpr std::uint64_t convert(unsigned n)
    { return supported(n) ? (std::uint64_t(3)*n+1)*(32*n+1076)+8*n+32:unavailable; }
    static constexpr std::uint64_t clear(unsigned n) { return supported(n) ? n+1:unavailable; }
};
inline bool widthSupported(unsigned n) { return n>=1 && n<=87; }
inline std::uint32_t floatBits(float v)
{ std::uint32_t bits; std::memcpy(&bits,&v,4); return bits; }
inline float floatFromBits(std::uint32_t bits)
{ float v; std::memcpy(&v,&bits,4); return v; }

inline bool clear(Integer& out,unsigned n)
{
    if (!widthSupported(n)) return false;
    for (unsigned i=0;i<n;++i) out.word[i]=0;
    out.negative=0; return true;
}
// For a nonzero decoded mantissa, five fixed bit-group tests suffice.
inline unsigned trailingZeros(std::uint32_t v)
{
    unsigned n=0;
    if (!(v&0xffffU)) { n+=16; v>>=16; }
    if (!(v&0xffU)) { n+=8; v>>=8; }
    if (!(v&0xfU)) { n+=4; v>>=4; }
    if (!(v&3U)) { n+=2; v>>=2; }
    return n+static_cast<unsigned>(!(v&1U));
}
inline bool decodeFloat(Integer& out,float v,unsigned n,unsigned grid)
{
    if (!widthSupported(n) || grid>149) return false;
    const auto bits=floatBits(v),absolute=bits&0x7fffffffU;
    const unsigned exponent=absolute>>23;
    if (absolute && (exponent==0 || exponent==255)) return false;
    const auto mantissa=absolute ? (absolute&0x7fffffU)|0x800000U:0;
    const auto trailing=absolute ? trailingZeros(mantissa):0;
    const int shift=absolute ? int(exponent)-150+int(trailing)+int(grid):0;
    if (shift<0) return false;
    const unsigned index=unsigned(shift)/32,offset=unsigned(shift)%32;
    const auto reduced=mantissa>>trailing;
    const auto high=offset ? reduced>>(32-offset):0;
    if (index>=n || (high && index+1>=n)) return false;
    // F has one n-word initialization loop and fixed bit/grid checks. The two
    // value stores replace already initialized words; no other reset occurs.
    for (unsigned i=0;i<n;++i) out.word[i]=0;
    out.word[index]=reduced<<offset;
    if (high) out.word[index+1]=high;
    out.negative=absolute ? bits>>31:0; return true;
}
inline bool compare(const Integer& a,const Integer& b,unsigned n,int& result)
{
    if (!widthSupported(n) || a.negative>1 || b.negative>1) return false;
    int magnitude=0; std::uint32_t anyA=0,anyB=0;
    // One n-word comparison/reduction pass also recognizes signed zero.
    for (unsigned i=0;i<n;++i) {
        anyA|=a.word[i]; anyB|=b.word[i];
        if (a.word[i]!=b.word[i]) magnitude=a.word[i]<b.word[i] ? -1:1;
    }
    const auto an=anyA ? a.negative:0,bn=anyB ? b.negative:0;
    result=an!=bn ? (an ? -1:1):(an ? -magnitude:magnitude);
    return true;
}
inline bool add(Integer& out,const Integer& a,const Integer& b,unsigned n,bool subtract=false)
{
    if (!widthSupported(n) || a.negative>1 || b.negative>1) return false;
    int magnitude=0; std::uint32_t anyA=0,anyB=0;
    // The reduction/comparison and result loops use 2n visits, below A=3n+8.
    // All sign decisions precede stores, so either source may alias out.
    for (unsigned i=0;i<n;++i) {
        anyA|=a.word[i]; anyB|=b.word[i];
        if (a.word[i]!=b.word[i]) magnitude=a.word[i]<b.word[i] ? -1:1;
    }
    const auto an=anyA ? a.negative:0;
    const auto bn=anyB ? b.negative^unsigned(subtract):0;
    const bool sum=an==bn,reverse=!sum && magnitude<0;
    const auto& left=reverse ? b:a;
    const auto& right=reverse ? a:b;
    std::uint64_t carry=0; std::uint32_t any=0;
    for (unsigned i=0;i<n;++i) {
        const std::uint64_t original=left.word[i];
        const std::uint64_t operand=std::uint64_t(right.word[i])+carry;
        const auto value=sum ? original+operand:original-operand;
        carry=sum ? value>>32:unsigned(original<operand);
        out.word[i]=std::uint32_t(value); any|=out.word[i];
    }
    out.negative=any ? (reverse ? bn:an):0;
    // Overflow leaves only primitive scratch unspecified; callers stage it.
    return !sum || !carry;
}
inline bool subtract(Integer& out,const Integer& a,const Integer& b,unsigned n)
{ return add(out,a,b,n,true); }
inline bool shiftLeft(Integer& out,const Integer& a,unsigned n,unsigned shift)
{
    if (!widthSupported(n) || a.negative>1) return false;
    const auto sign=a.negative;
    const unsigned whole=shift/32,partial=shift%32;
    std::uint32_t any=0,overflow=0;
    // One descending n-word pass checks discarded bits and constructs output.
    // Lower source words have not been overwritten when out aliases a.
    for (unsigned i=n;i--;) {
        const auto original=a.word[i];
        if (whole>=n || i>=n-whole) overflow|=original;
        else if (partial && i==n-whole-1) overflow|=original>>(32-partial);
        const auto value=i<whole ? 0:
            (a.word[i-whole]<<partial)
                |(partial && i>whole ? a.word[i-whole-1]>>(32-partial):0);
        out.word[i]=value; any|=value;
    }
    out.negative=any ? sign:0; return overflow==0;
}
inline bool multiply(Integer& out,const Integer& a,const Integer& b,unsigned n,Workspace& workspace)
{
    if (!widthSupported(n) || a.negative>1 || b.negative>1) return false;
    auto& product=workspace.product;
    // M: 2n zero visits, n*n multiply cells, n carry stores, n overflow
    // inspections and n result stores. No Integer or workspace copy exists.
    for (unsigned i=0;i<2*n;++i) product[i]=0;
    for (unsigned i=0;i<n;++i) {
        std::uint64_t carry=0;
        for (unsigned j=0;j<n;++j) {
            const auto cell=std::uint64_t(a.word[i])*b.word[j]+product[i+j]+carry;
            product[i+j]=std::uint32_t(cell); carry=cell>>32;
        }
        product[i+n]=std::uint32_t(carry);
    }
    std::uint32_t overflow=0;
    for (unsigned i=n;i<2*n;++i) overflow|=product[i];
    if (overflow) return false;
    const auto sign=a.negative^b.negative;
    std::uint32_t any=0;
    for (unsigned i=0;i<n;++i) { out.word[i]=product[i]; any|=out.word[i]; }
    out.negative=any ? sign:0; return true;
}

// remainder must be a distinct admitted Integer, borrowed from the caller's
// existing register set. A pointer-alias refusal precedes its first write.
// Failure never updates the caller's converted bits or rounding direction.
inline bool convertBits(const Integer& numerator,const Integer& denominator,unsigned n,
    bool binary64,Integer& remainder,std::uint64_t& output,bool& roundedAbove)
{
    if (!widthSupported(n) || numerator.negative>1 || denominator.negative>1
        || &remainder==&numerator || &remainder==&denominator) return false;
    std::uint32_t anyN=0,anyD=0;
    for (unsigned i=0;i<n;++i) { anyN|=numerator.word[i]; anyD|=denominator.word[i]; }
    if (!anyD) return false;
    if (!anyN) { output=0; roundedAbove=false; return true; }
    for (unsigned i=0;i<n;++i) remainder.word[i]=0;
    remainder.negative=0;
    const unsigned precision=binary64 ? 53:24,signBit=binary64 ? 63:31;
    const int minimum=binary64 ? -1022:-126,maximum=binary64 ? 1023:127;
    const int bias=binary64 ? 1023:127;
    const auto sign=numerator.negative^denominator.negative;
    std::uint64_t significand=0;
    bool seen=false,guard=false;
    int exponent=0,quantum=minimum-int(precision)+1;
    int position=int(32*n)-1;
    // At most 32n+1076 bit transitions. Each has <=n shift visits, <=n
    // magnitude comparisons and <=n subtract visits. Its scalar transition
    // updates quotient/leading-bit/rounding state; this is R's +1 per bit.
    for (;position>=-1076;--position) {
        std::uint32_t carry=position>=0
            ? (numerator.word[unsigned(position)/32]>>(position%32))&1U:0;
        for (unsigned i=0;i<n;++i) {
            const auto previous=remainder.word[i];
            remainder.word[i]=(previous<<1)|carry; carry=previous>>31;
        }
        int magnitude=0;
        for (unsigned i=n;i-- && !magnitude;)
            if (remainder.word[i]!=denominator.word[i])
                magnitude=remainder.word[i]<denominator.word[i] ? -1:1;
        const bool quotient=carry || magnitude>=0;
        if (quotient) {
            std::uint64_t borrow=0;
            for (unsigned i=0;i<n;++i) {
                const auto sub=std::uint64_t(denominator.word[i])+borrow;
                const auto previous=std::uint64_t(remainder.word[i]);
                remainder.word[i]=std::uint32_t(previous-sub); borrow=previous<sub;
            }
            if (carry!=borrow) return false;
        }
        if (!seen && quotient) {
            seen=true; exponent=position;
            if (exponent>maximum) return false;
            quantum=(exponent<minimum ? minimum:exponent)-int(precision)+1;
        }
        if (position>=quantum && seen) significand=(significand<<1)|unsigned(quotient);
        if (position==quantum-1) { guard=quotient; break; }
    }
    std::uint32_t remaining=0;
    for (unsigned i=0;i<n;++i) remaining|=remainder.word[i];
    bool sticky=remaining!=0;
    // At most n unread-source visits. No native float/divide/libm is an oracle.
    if (position>0) for (unsigned i=0;i<n;++i) {
        const int lower=int(i*32);
        if (lower>=position) break;
        const unsigned available=unsigned(position-lower>32 ? 32:position-lower);
        const auto mask=available==32 ? 0xffffffffU:(std::uint32_t(1)<<available)-1;
        sticky=sticky || (numerator.word[i]&mask)!=0;
    }
    const bool increment=guard && (sticky || (significand&1));
    const bool above=sign ? (!increment && (guard || sticky)):increment;
    if (increment) ++significand;
    std::uint64_t result;
    if (seen && exponent>=minimum) {
        if (significand==(std::uint64_t(1)<<precision)) { significand>>=1; ++exponent; }
        if (exponent>maximum) return false;
        result=(std::uint64_t(sign)<<signBit)|(std::uint64_t(exponent+bias)<<(precision-1))
            |(significand&((std::uint64_t(1)<<(precision-1))-1));
    } else result=(std::uint64_t(sign)<<signBit)|significand;
    output=result; roundedAbove=above; return true;
}
inline bool convertFloat(const Integer& numerator,const Integer& denominator,unsigned n,
    Integer& remainder,float& output,bool& roundedAbove)
{
    std::uint64_t bits; bool above;
    if (!convertBits(numerator,denominator,n,false,remainder,bits,above)) return false;
    const auto narrow=std::uint32_t(bits);
    std::memcpy(&output,&narrow,4); roundedAbove=above; return true;
}
inline bool convertDouble(const Integer& numerator,const Integer& denominator,unsigned n,
    Integer& remainder,double& output,bool& roundedAbove)
{
    std::uint64_t bits; bool above;
    if (!convertBits(numerator,denominator,n,true,remainder,bits,above)) return false;
    std::memcpy(&output,&bits,8); roundedAbove=above; return true;
}
}
