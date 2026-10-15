#pragma once
#include "../../src/gpu_backend/fixture_native_lambertian_radiance.h"
#include <array>
#include <cstdint>

namespace arbitgpu::lambertiantests
{
namespace lr=lambertian;
// Prepared only. No existing oracle, executable, native owner or renderer hook.
// Future exclusive Metal fixture join must dispatch metalArithmeticSource's
// lrEvaluate on authenticated actual readback. Dispatch(q,result) returns the
// entry's bool, including untouched output on insufficient native admission.
struct Case {
    const char* name;
    lr::Request request;
    bool supported;
    bool fixedBits;
    std::uint32_t bits;
};
inline lr::Request request(std::uint64_t significand,int exponent)
{
    lr::Request q{};q[0]=lr::kVersion;q[1]=lr::kP32Bits;q[2]=128;q[3]=640;
    if(significand){for(unsigned k=0;k<64;k++)if((significand&1u)==0){significand>>=1;exponent++;}}
    else exponent=0;
    q[4]=std::uint32_t(exponent);q[5]=std::uint32_t(significand);q[6]=std::uint32_t(significand>>32);
    lr::putBudgets(q,lr::kNativeCeilings);return q;
}
// These are source ceilings for prepared arithmetic checks only. Driver,
// compiled spills/pipelines, buffers, retained owners and transfer admission
// remain additional. This supplies no complete native attempt reservation.
constexpr lr::Ceilings kPreparationCeilings{2*lr::kHostCeilings.limbVisits+100000,
    2*lr::kHostCeilings.wordMoves+100000,2*lr::kHostCeilings.scalarOperations+1000000,
    2*lr::kHostCeilings.parserOperations+100000,2*lr::kHostCeilings.comparisons+1000000,
    lr::kHostCeilings.scratchBytes+16000};
// 24 deterministic pairs,6 short native dispatches,1 full-precision and2
// canonical mutations =57 dispatches. 24 ordinary+246 certificate mutations+
// 6 short host admissions+1 precision+2 canonical mutations =279 inspections.
constexpr lr::Ceilings kPreparedChecksCeilings{
    kPreparationCeilings.limbVisits+57*lr::kNativeCeilings.limbVisits+279*lr::kHostCeilings.limbVisits,
    kPreparationCeilings.wordMoves+57*lr::kNativeCeilings.wordMoves+279*lr::kHostCeilings.wordMoves+1000000,
    kPreparationCeilings.scalarOperations+57*lr::kNativeCeilings.scalarOperations+279*lr::kHostCeilings.scalarOperations+10000000,
    kPreparationCeilings.parserOperations+57*lr::kNativeCeilings.parserOperations+279*lr::kHostCeilings.parserOperations+100000,
    kPreparationCeilings.comparisons+57*lr::kNativeCeilings.comparisons+279*lr::kHostCeilings.comparisons+10000000,
    kPreparationCeilings.scratchBytes+lr::kNativeCeilings.scratchBytes+lr::kHostCeilings.scratchBytes+16000};
inline lr::Request physicalBoundary(bool above,bool& valid,const lr::Ceilings& allowance)
{
    // Consecutive dyadics around65504*piL. PiL has odd denominator factors, so
    // equality with this2^-64 grid is impossible. Retain/check actual remainder.
    if(!lr::admitted(allowance,kPreparationCeilings)){valid=false;return {}; }
    lr::host::Work w;const auto p=lr::host::pi(w);
    const auto numerator=lr::host::shift(lr::host::scale(p.lower,65504,w),64,w);
    auto division=lr::host::divide(numerator,p.denominator,w);
    valid=w.ok&&!lr::host::zero(division.remainder);
    if(above)division.quotient=lr::host::add(division.quotient,lr::host::small(1),w);
    int exponent=-64;
    for(unsigned round=0;round<128;round++)if(!lr::host::zero(division.quotient)&&(division.quotient.word[0]&1u)==0){
        // Bounded128-bit canonicalization; boundary numerator fits82 bits.
        for(unsigned k=0;k<4;k++)division.quotient.word[k]=(division.quotient.word[k]>>1)
            |(k+1<4?division.quotient.word[k+1]<<31:0u);
        exponent++;
    }
    auto q=request(1,0);q[4]=std::uint32_t(exponent);
    for(unsigned k=0;k<4;k++)q[5+k]=division.quotient.word[k];
    for(unsigned k=4;k<lr::kLimbs;k++)if(division.quotient.word[k])valid=false;
    valid=valid&&w.ok;return q;
}
inline std::array<Case,24> cases(bool& valid,const lr::Ceilings& allowance)
{
    if(!lr::admitted(allowance,kPreparationCeilings)){valid=false;return {}; }
    const std::uint64_t p=lr::kP32Significand;
    bool lowerValid=false,upperValid=false;
    const auto below=physicalBoundary(false,lowerValid,allowance),above=physicalBoundary(true,upperValid,allowance);
    valid=lowerValid&&upperValid;
    std::array<Case,24> rows{{
        {"exact zero",request(0,0),true,true,0},
        {"exact one",request(p,-22),true,true,0x3f800000u},
        {"next normal",request(p*8388609ull,-45),true,true,0x3f800001u},
        {"even lower midpoint",request(p*16777217ull,-46),true,true,0x3f800000u},
        {"even upper midpoint",request(p*16777219ull,-46),true,true,0x3f800002u},
        {"below lower midpoint",request(p*16777217ull-1,-46),true,true,0x3f800000u},
        {"above lower midpoint",request(p*16777217ull+1,-46),true,true,0x3f800001u},
        {"binade carry midpoint",request(p*33554431ull,-46),true,true,0x40000000u},
        {"minimum normal",request(p,-148),true,true,0x00800000u},
        {"minimum normal tie",request(p*16777215ull,-172),true,true,0x00800000u},
        {"below normal tie",request(p*16777215ull-1,-172),false,false,0},
        {"positive subnormal refused",request(p,-149),false,false,0},
        {"physical upper below",below,true,false,0},
        {"physical upper above",above,false,false,0},
        {"minimum allowed exponent",request(1,-253),false,false,0},
        {"maximum allowed exponent",request(1,17),true,false,0},
        {"exponent below",request(1,-254),false,false,0},
        {"exponent above",request(1,18),false,false,0},
        {"physical overflow",request(3,17),false,false,0},
        {"zero exponent mutation",request(0,0),false,false,0},
        {"even significand mutation",request(p,-22),false,false,0},
        {"precision mutation",request(p,-22),false,false,0},
        {"capacity mutation",request(p,-22),false,false,0},
        {"P32 mutation",request(p,-22),false,false,0}
    }};
    rows[19].request[4]=1;rows[20].request[5]&=~1u;rows[21].request[2]=129;
    rows[22].request[3]=641;rows[23].request[1]^=1u;return rows;
}
// No native execution is performed by including this header. The future owner
// runs this function only inside its separately admitted exact candidate lane.
// Harness/case preparation, source assembly, transfers and driver/compiler/GPU
// allocations need their own admission; these helpers inherit none of it.
// Arithmetic verification must use full native and host ceilings per attempted
// dispatch/inspection, including every mutation and deterministic repeat below.
template<class Dispatch>
inline bool runPrepared(Dispatch&& dispatch,const lr::Ceilings& allowance)
{
    if(!lr::admitted(allowance,kPreparedChecksCeilings))return false;
    bool valid=false;const auto rows=cases(valid,kPreparationCeilings);if(!valid)return false;
    lr::Result positive{};lr::Request positiveRequest{};bool retained=false;
    for(const auto& row:rows){
        lr::Result result{},repeat{};
        if(!dispatch(row.request,result)||!dispatch(row.request,repeat)||result!=repeat)return false;
        const auto outcome=lr::inspect(row.request,result,lr::kHostCeilings);
        if(outcome!=(row.supported?lr::Inspection::Certified:lr::Inspection::Refused))return false;
        if(row.supported&&row.fixedBits&&result[2]!=row.bits)return false;
        if(row.fixedBits&&row.bits==0x3f800000u&&!retained){positive=result;positiveRequest=row.request;retained=true;}
    }
    if(!retained)return false;
    // Every certificate word is checked. Includes P32 echo, precision, pi D/L/U,
    // quotient/remainder/denominator, neighbor-certified bits, interval and error.
    for(unsigned word=0;word<lr::kResultWords;word++){
        auto mutated=positive;mutated[word]^=1u;
        if(lr::inspect(positiveRequest,mutated,lr::kHostCeilings)!=lr::Inspection::Invalid)return false;
    }
    // Shorten each independent native category, require pre-work refusal with
    // sentinel output preserved. Also shorten each host category before parsing.
    for(unsigned category=0;category<6;category++){
        auto q=positiveRequest;const unsigned lo=10+2*category;
        if(q[lo]==0){q[lo]=0xffffffffu;q[lo+1]--;}else q[lo]--;
        lr::Result untouched;untouched.fill(0xa5a5a5a5u);const auto before=untouched;
        if(dispatch(q,untouched)||untouched!=before)return false;
        auto hostAllowance=lr::kHostCeilings;
        switch(category){case 0:hostAllowance.limbVisits--;break;case 1:hostAllowance.wordMoves--;break;
            case 2:hostAllowance.scalarOperations--;break;case 3:hostAllowance.parserOperations--;break;
            case 4:hostAllowance.comparisons--;break;default:hostAllowance.scratchBytes--;break;}
        if(lr::inspect(positiveRequest,positive,hostAllowance)!=lr::Inspection::NotAdmitted)return false;
    }
    // Full128-bit precision has a genuine normal positive output.
    auto high=request(1,-251);high[5]=high[6]=high[7]=high[8]=0xffffffffu;
    lr::Result result{};if(!dispatch(high,result)||lr::inspect(high,result,lr::kHostCeilings)!=lr::Inspection::Certified)return false;
    // 128 significant bits at2^-251 is about2^-123, giving normal M/P32.
    // Reserved and version fields are separate canonical refusal controls.
    for(unsigned at:std::array<unsigned,2>{{0,9}}){auto q=positiveRequest;q[at]^=1u;
        result={};if(!dispatch(q,result)||lr::inspect(q,result,lr::kHostCeilings)!=lr::Inspection::Refused)return false;}
    return true;
}
} // namespace arbitgpu::lambertiantests
