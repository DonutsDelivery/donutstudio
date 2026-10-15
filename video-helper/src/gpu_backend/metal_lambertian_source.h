#pragma once
#include "fixture_native_lambertian_radiance.h"
#include <array>
#include <cstddef>
#include <cstring>

namespace arbitgpu::lambertian::native
{
constexpr std::uint32_t kSourceVersion=1,kEntryMarker=0x4c523031u;
inline constexpr char kMetalPrefix[]="#include <metal_stdlib>\nusing namespace metal;\n";
inline constexpr char kPrefix[]="#include <metal_stdlib>\nusing namespace metal;\n"
    "#define EL_REF(T,N) thread T& N\n#define EL_VAL(T,N) T N\n"
    "#define EL_REQUEST(N) thread const uint* N\n#define EL_RESULT(N) thread uint* N\n";
inline constexpr char kBoundStart[]="uint elBound(uint op){return op==1u?";
inline constexpr char kBoundEnd[]="u:0u;}\n#define EL_BOUND(O) elBound(O)\n";
inline constexpr char kLrStart[]="\n#define LR_REQUEST(N) thread const uint (&N)[22]\n#define LR_RESULT(N) thread uint (&N)[246]\n#define LR_VISITS ";
inline constexpr char kAdmitStart[]="u\nbool lrAdmit(LR_REQUEST(q)){\n";
inline constexpr char kAdmitEnd[]="return true;}\n#define LR_ADMIT(Q) lrAdmit(Q)\n";
// 22 input, 246 sentinel, 246 result, four trailer and four uniform word
// moves. Each loop iteration has at most eight index/control/statement
// operations. Guards, call and trailer statements add at most64 operations.
// Arithmetic inside lrEvaluate has its separate accepted census.
inline constexpr char kEntry[]=R"ENTRY(
kernel void lambertianArithmetic(device const uint* input [[buffer(0)]],
    device uint* output [[buffer(1)]],constant uint* limits [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if(tid!=0u)return;
    if(limits[0]!=22u||limits[1]!=246u)return;
    uint q[22];uint r[246];
    for(uint k=0u;k<22u;k++)q[k]=input[k];
    for(uint k=0u;k<246u;k++)r[k]=output[k];
    bool evaluated=lrEvaluate(q,r);
    for(uint k=0u;k<246u;k++)output[k]=r[k];
    output[246]=0x4c523031u;output[247]=evaluated?1u:0u;
    output[248]=limits[2];output[249]=limits[3];
}
)ENTRY";
constexpr std::size_t kRawBytes=6401+6078,kDecimalCalls=42,kDecimalBytes=276;
// Five EL constants, LR_VISITS, then six clauses with three indices and three
// uint32 values. Prefixes and suffixes are counted once per actual append.
constexpr std::size_t kFixedBytes=sizeof(kPrefix)-1+sizeof(kBoundStart)-1
    +4*(sizeof("u:op==2u?")-1)+(sizeof("u:op==5u||op==6u?")-1)-(sizeof("u:op==2u?")-1)
    +sizeof(kBoundEnd)-1+sizeof(kLrStart)-1+sizeof(kAdmitStart)-1+sizeof(kAdmitEnd)-1
    +6*((sizeof("if(q[")-1)+(sizeof("]<")-1)+(sizeof("u||(q[")-1)+(sizeof("]==")-1)
        +(sizeof("u&&q[")-1)+(sizeof("]<")-1)+(sizeof("u))return false;\n")-1))
    +sizeof(kEntry)-1;
constexpr std::size_t kSourceCapacity=kRawBytes+kFixedBytes+kDecimalBytes+1;
constexpr std::uint64_t kSourceVisits=2*kSourceCapacity+10*kDecimalCalls;
constexpr std::uint64_t kSourceMoves=kSourceCapacity+20*kDecimalCalls;
constexpr std::uint64_t kSourceScalars=16*kSourceVisits+16*kDecimalCalls;
constexpr std::uint64_t kWrapperWords=522,kWrapperScalars=8*(22+2*246)+64;
struct Source {
    std::array<char,kSourceCapacity> bytes{};
    std::size_t size=0,arithmeticOffset=sizeof(kMetalPrefix)-1,arithmeticBytes=0;
    std::uint32_t decimalCalls=0;
};
// Caller admits all source/output/copy capacities and this complete schedule
// before construction. No allocating string builder or arithmetic executes.
inline bool assemble(Source& out)
{
    out.size=out.decimalCalls=out.arithmeticBytes=0;
    out.arithmeticOffset=sizeof(kMetalPrefix)-1;
    const auto append=[&](const char* text){const auto count=std::strlen(text);
        if(count>=out.bytes.size()-out.size)return false;
        std::memcpy(out.bytes.data()+out.size,text,count);out.size+=count;out.bytes[out.size]=0;return true;};
    const auto decimal=[&](std::uint32_t value){char digits[10];unsigned count=0;
        do{digits[count++]=char('0'+value%10);value/=10;}while(value);
        if(count>=out.bytes.size()-out.size)return false;
        while(count)out.bytes[out.size++]=digits[--count];
        out.bytes[out.size]=0;++out.decimalCalls;return true;};
    using exactline::Operation;
    if(!append(kPrefix)||!append(kBoundStart)||!decimal(exactline::maximumVisits(Operation::Det2))
        ||!append("u:op==2u?")||!decimal(exactline::maximumVisits(Operation::Det3))
        ||!append("u:op==3u?")||!decimal(exactline::maximumVisits(Operation::Cross))
        ||!append("u:op==4u?")||!decimal(exactline::maximumVisits(Operation::Incidence))
        ||!append("u:op==5u||op==6u?")||!decimal(exactline::maximumVisits(Operation::CompareX))
        ||!append(kBoundEnd)||std::strlen(exactline::arithmeticSource())!=6401
        ||!append(exactline::arithmeticSource())||!append(kLrStart)||!decimal(std::uint32_t(kNativeVisits))
        ||!append(kAdmitStart))return false;
    const std::uint64_t values[]{kNativeCeilings.limbVisits,kNativeCeilings.wordMoves,kNativeCeilings.scalarOperations,
        kNativeCeilings.parserOperations,kNativeCeilings.comparisons,kNativeCeilings.scratchBytes};
    for(unsigned i=0;i<6;i++)if(!append("if(q[")||!decimal(11+2*i)||!append("]<")||!decimal(std::uint32_t(values[i]>>32))
        ||!append("u||(q[")||!decimal(11+2*i)||!append("]==")||!decimal(std::uint32_t(values[i]>>32))
        ||!append("u&&q[")||!decimal(10+2*i)||!append("]<")||!decimal(std::uint32_t(values[i]))
        ||!append("u))return false;\n"))return false;
    if(!append(kAdmitEnd)||std::strlen(arithmeticSource())!=6078||!append(arithmeticSource()))return false;
    out.arithmeticBytes=out.size-out.arithmeticOffset;
    return append(kEntry)&&out.decimalCalls==kDecimalCalls;
}
} // namespace arbitgpu::lambertian::native
