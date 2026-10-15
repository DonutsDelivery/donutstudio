#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace arbitgpu::exactline
{
// Arithmetic only, on already-formed binary32 coefficients. This is neither a
// camera/scene association nor a native visibility/coverage certificate. The
// association of rounded camera coefficients with frozen query/raster resources
// remains unresolved. No ideal algebraic camera is substituted here.
enum class Operation : std::uint32_t { Det2=1, Det3=2, Cross=3, Incidence=4, CompareX=5, CompareY=6 };
enum class Status : std::uint32_t { Unsupported=0, Exact=1 };
enum class Classification : std::uint32_t { None=0, Finite=1, Parallel=2, Coincident=3 };
// Signs use 0=zero, 1=positive, 2=negative (including rational comparisons).
constexpr std::uint32_t kCoefficientLimbs=9, kDet2Limbs=18, kDet3Limbs=27, kComparisonLimbs=35;
constexpr std::uint32_t kRequestWords=16, kResultWords=62;
using Request=std::array<std::uint32_t,kRequestWords>;
using Result=std::array<std::uint32_t,kResultWords>;
static_assert(sizeof(Request)==64&&sizeof(Result)==248,"raw uint32 probe ABI");
// Request: operation, admitted limb-visit limit, capacity (1..35), reserved=0,
// then 12 raw coefficient words. Det2 consumes four; Det3/Incidence nine;
// Cross six; comparisons four lines (12). Unused words must be zero.
// Result: status, sign, classification, admitted visits, performed visits,
// followed by three signed 18-limb cross coordinates (sign then low limbs).
// Refusal zeros all arithmetic output; admitted/performed visits remain.
// The zero line is refused. Parallel means a nonzero cross with z=0 (an
// infinite intersection, also for a supplied projective line at infinity).
// Cross classification may be Coincident, but its zero vector is not a point.
// Incidence and ordering refuse degenerate points; ordering also refuses z=0.
// Equal rational coordinates return zero, never a fabricated tie order.

// Every finite f32 is I*2^-149, |I|<2^277 (9 limbs). A det2 is <2^555
// (18 limbs). Laplace det3 intermediate totals are <6*2^831<2^834
// (27 limbs). A difference of two det2-coordinate products is <2^1111
// (35 limbs). These include intermediate signed sums, not just final values.
// Fixed storage is 144 bytes/integer. Evaluate has 12 decoded coefficients,
// six cross coordinates, nested locals/value parameters and ternary temporaries: <=40*144
// =5760 logical bytes, plus request/result/counters. Shader compiler spills,
// program/pipeline and dispatch buffers MUST be charged separately by a future
// native owner before work. These helpers authenticate no owner or parent.
constexpr std::uint64_t kIntegerBytes=4*(kComparisonLimbs+1);
constexpr std::uint64_t kLogicalScratchBytes=40*kIntegerBytes+kRequestWords*4+kResultWords*4+64;
// Limb visits: multiplication <=5*35+(1+2*35)*a*b (clear, two input
// capacity scans, normalize, result scan; two full carry walks per product).
// Signed addition <=5*35. Decode <=35+2. Each det2 adds one 35-limb
// result scan; det3/incidence add two. These bounds charge failed work too.
constexpr std::uint32_t multiplyVisits(std::uint32_t a,std::uint32_t b){return 175+71*a*b;}
constexpr std::uint32_t kDet2Visits=2*multiplyVisits(9,9)+175+35;
constexpr std::uint32_t kCrossVisits=3*kDet2Visits;
constexpr std::uint32_t kDet3Visits=kCrossVisits+3*multiplyVisits(9,18)+2*175+2*35;
constexpr std::uint32_t kCompareVisits=2*kCrossVisits+2*multiplyVisits(18,18)+175;
constexpr std::uint32_t maximumVisits(Operation operation)
{
    return operation==Operation::Det2?4*37+kDet2Visits:
        operation==Operation::Det3?9*37+kDet3Visits:
        operation==Operation::Cross?6*37+kCrossVisits:
        operation==Operation::Incidence?9*37+kCrossVisits+3*multiplyVisits(9,18)+2*175+2*35:
        operation==Operation::CompareX||operation==Operation::CompareY?12*37+kCompareVisits:0;
}
static_assert(maximumVisits(Operation::CompareX)==119349,"derived comparison admission");
static_assert(kLogicalScratchBytes==6136,"logical storage, excluding compiler/native owners");

// One shared body. The only language differences are reference/address-space
// syntax and the test dispatch wrapper. No float operations or shader int64.
// Split16 multiplication: t<=0xffff0000 and v<=0xffff0000; all four
// 16x16 products and monotone high-word sums fit uint32. Addition/subtraction
// use 16-bit halves, whose intermediates are <=131071; no ordinary uint
// multiplication/addition overflows. GLSL 4.30 sections 4.1.3/8.8:
// https://registry.khronos.org/OpenGL/specs/gl/GLSLangSpec.4.30.pdf
// Prospective source-cost envelope, not measured GPU operations or a lower bound.
// Reserve64 per worst-case visit for scalar arithmetic, loop/index costs and
// amortized EL setup. A sum's35 arithmetic iterations may each exceed64; its
// other140 reserved clear/scan iterations make the complete175-visit bound safe.
// Excluding separately charged copies, arithmetic iterations cost<=100 each
// and clear/scan iterations<=32:35*100+140*32=7980 <=64*175=11200.
// Nested copies are separate, including GLSL full value/inout copies even if
// optimized away. Metal fits the same envelope. Each moved word reserves8 for
// move/loop/index costs; this is a conservative multiplier, not an instruction count.
// ELInt=36 words, ELWork=4. Call moves: decode81, cross440, det2224,
// multiply155, sum152, zero80, capacity45, normalize80, magnitude81,
// accumulate82, step8, product<=8, word helper<=6. Every product may attempt
// two full35-word carry walks; each sum has two word helpers per35-word iteration.
// Fixed4096 covers wrapper78+entry78+clear62+cross output57+metadata5+work4
// +two ternary integer temporaries72=356 moves, plus fixed scalar/return moves.
// Setup32768+12*128 covers entry checks/branches and all straight-line decodes.
// Saturated counters do not stop all loops: charge the full attempted schedule.
constexpr std::uint64_t maximumWordMoves(Operation operation)
{
    if(maximumVisits(operation)==0)return 4096;
    const bool det2=operation==Operation::Det2;
    const bool tail=operation==Operation::Det3||operation==Operation::Incidence;
    const bool compare=operation==Operation::CompareX||operation==Operation::CompareY;
    const std::uint64_t n=det2?4:tail?9:operation==Operation::Cross?6:12;
    const std::uint64_t c=det2?0:compare?2:1,d=det2?1:3*c,t=tail?1:0,v=compare?1:0;
    const auto m=2*d+3*t+2*v,s=d+2*t+v,b=162*d+486*t+648*v;
    // Expanded schedule:81n+440c+224d+155m+152s+80(n+m+s)
    // +45(3m+s+d+2t)+80(m+s)+81s+164b+8V+8b+6(70b+70s).
    return 4096+161*n+440*c+269*d+450*m+858*s+90*t+592*b+8*maximumVisits(operation);
}
constexpr std::uint64_t maximumScalarOperations(Operation operation)
{return 34304+64ull*maximumVisits(operation)+8*maximumWordMoves(operation);}
static_assert(maximumWordMoves(Operation::CompareX)==1934660
    &&maximumScalarOperations(Operation::CompareX)==23149920,"raw complete comparison cost schedule");
// Invalid opcodes still run output clears. Physical scalar admission is positive;
// their unchanged mathematical/visit wire remains the all-zero refusal.
static_assert(maximumScalarOperations(static_cast<Operation>(0))==67072,"invalid raw dispatch setup cost");
// Transfers, host/caller copies, actual compiler/spill/pipeline storage and retained
// native owners are additional. No conversion to geometry-query-work is asserted.

inline const char* arithmeticSource()
{
    return R"EXACT(
#define EL_LIMBS 35
struct ELInt { uint limb[EL_LIMBS]; uint sign; };
struct ELWork { uint used; uint limit; uint capacity; bool ok; };
void elStep(EL_REF(ELWork,w)) { if(w.used>=w.limit){w.ok=false;return;} w.used+=1u; }
void elZero(EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    r.sign=0u;for(int i=0;i<35;i++){elStep(w);r.limb[i]=0u;}
}
void elNormalize(EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    bool anyWord=false;for(int i=0;i<35;i++){elStep(w);anyWord=anyWord||r.limb[i]!=0u;}
    if(!anyWord)r.sign=0u;
}
void elCapacity(EL_VAL(ELInt,a),uint cap,EL_REF(ELWork,w)) {
    for(int i=0;i<35;i++){elStep(w);if(uint(i)>=cap&&a.limb[i]!=0u)w.ok=false;}
}
uint elAddWord(uint a,uint b,EL_REF(uint,carry)) {
    uint lo=(a&65535u)+(b&65535u);
    uint hi=(a>>16u)+(b>>16u)+(lo>>16u);
    carry=hi>>16u;return (hi<<16u)|(lo&65535u);
}
uint elSubWord(uint a,uint b,EL_REF(uint,borrow)) {
    uint al=a&65535u,bl=b&65535u;
    uint low=al+65536u-bl;uint lowBorrow=al<bl?1u:0u;
    uint ah=a>>16u,bh=b>>16u;
    uint high=ah+65536u-bh-lowBorrow;
    borrow=ah<bh+lowBorrow?1u:0u;
    return (high<<16u)|(low&65535u);
}
void elProduct(uint a,uint b,EL_REF(uint,high),EL_REF(uint,low)) {
    uint a0=a&65535u,a1=a>>16u,b0=b&65535u,b1=b>>16u;
    uint p=a0*b0,t=a1*b0+(p>>16u),v=a0*b1+(t&65535u);
    high=a1*b1+(t>>16u)+(v>>16u);low=(v<<16u)|(p&65535u);
}
void elAccumulate(EL_REF(ELInt,r),int at,uint value,EL_REF(ELWork,w)) {
    for(int k=0;k<35;k++) {
        elStep(w);if(value==0u)break;
        if(at+k>=35){w.ok=false;return;}
        uint carry=0u;r.limb[at+k]=elAddWord(r.limb[at+k],value,carry);value=carry;
    }
    if(value!=0u)w.ok=false;
}
int elMagnitude(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_REF(ELWork,w)) {
    int result=0;for(int k=34;k>=0;k--){elStep(w);
        if(result==0&&a.limb[k]!=b.limb[k])result=a.limb[k]>b.limb[k]?1:-1;}
    return result;
}
void elSum(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    elZero(r,w);int cmp=elMagnitude(a,b,w);uint carry=0u;
    bool add=a.sign==b.sign;bool aFirst=cmp>=0;
    r.sign=add?a.sign:(aFirst?a.sign:b.sign);
    for(int k=0;k<35;k++) {
        elStep(w);uint next=0u,extra=0u;
        uint x=add||aFirst?a.limb[k]:b.limb[k];uint y=add||aFirst?b.limb[k]:a.limb[k];
        uint part=add?elAddWord(x,y,next):elSubWord(x,y,next);
        r.limb[k]=add?elAddWord(part,carry,extra):elSubWord(part,carry,extra);
        carry=next|extra;
    }
    if(carry!=0u)w.ok=false;elNormalize(r,w);elCapacity(r,w.capacity,w);
}
void elMultiply(EL_VAL(ELInt,a),EL_VAL(ELInt,b),uint na,uint nb,uint cap,
    EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    elZero(r,w);elCapacity(a,na,w);elCapacity(b,nb,w);
    r.sign=a.sign==0u||b.sign==0u?0u:(a.sign==b.sign?1u:2u);
    for(int i=0;i<int(na);i++)for(int j=0;j<int(nb);j++) {
        elStep(w);uint high=0u,low=0u;elProduct(a.limb[i],b.limb[j],high,low);
        elAccumulate(r,i+j,low,w);elAccumulate(r,i+j+1,high,w);
    }
    elNormalize(r,w);elCapacity(r,min(cap,w.capacity),w);
}
void elDecode(uint bits,EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    elZero(r,w);uint e=(bits>>23u)&255u,frac=bits&8388607u;
    if(e==255u){w.ok=false;return;}
    uint significand=e==0u?frac:(frac|8388608u),shift=e==0u?0u:e-1u;
    if(significand==0u)return;
    r.sign=(bits>>31u)==0u?1u:2u;uint at=shift/32u,offset=shift%32u;
    elStep(w);r.limb[at]=significand<<offset;
    if(offset!=0u){elStep(w);r.limb[at+1u]=significand>>(32u-offset);}
    if(at>=w.capacity||(offset!=0u&&at+1u>=w.capacity&&r.limb[at+1u]!=0u))w.ok=false;
}
void elDet2(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_VAL(ELInt,c),EL_VAL(ELInt,d),
    EL_REF(ELInt,r),EL_REF(ELWork,w)) {
    ELInt p,q;elMultiply(a,d,9u,9u,18u,p,w);elMultiply(b,c,9u,9u,18u,q,w);
    if(q.sign!=0u)q.sign=3u-q.sign;elSum(p,q,r,w);elCapacity(r,18u,w);
}
void elCross(EL_VAL(ELInt,a),EL_VAL(ELInt,b),EL_VAL(ELInt,c),
    EL_VAL(ELInt,d),EL_VAL(ELInt,e),EL_VAL(ELInt,f),
    EL_REF(ELInt,x),EL_REF(ELInt,y),EL_REF(ELInt,z),EL_REF(ELWork,w)) {
    elDet2(b,c,e,f,x,w);elDet2(c,a,f,d,y,w);elDet2(a,b,d,e,z,w);
}
// Explicit result words keep ABI identical in std430 and Metal uint buffers.
void elEvaluate(EL_REQUEST(requestWords),EL_RESULT(resultWords)) {
    for(int k=0;k<62;k++)resultWords[k]=0u;
    uint operation=requestWords[0],required=EL_BOUND(operation);
    if(required==0u||requestWords[1]<required||requestWords[2]==0u||requestWords[2]>35u||requestWords[3]!=0u)return;
    resultWords[3]=required;ELWork w;w.used=0u;w.limit=required;w.capacity=requestWords[2];w.ok=true;
    int count=operation==1u?4:(operation==2u||operation==4u?9:(operation==3u?6:12));
    ELInt a[12];for(int k=0;k<count;k++)elDecode(requestWords[k+4],a[k],w);
    for(int k=count;k<12;k++)if(requestWords[k+4]!=0u)w.ok=false;
    ELInt x,y,z,p,q,r,s,t;uint classification=0u,sign=0u;
    if(w.ok&&operation==1u){elDet2(a[0],a[1],a[2],a[3],r,w);sign=r.sign;}
    if(w.ok&&operation!=1u) {
        if(operation!=2u)for(int k=0;k<count;k+=3)
            if(a[k].sign==0u&&a[k+1].sign==0u&&a[k+2].sign==0u)w.ok=false;
        if(w.ok) {
            elCross(a[0],a[1],a[2],a[3],a[4],a[5],x,y,z,w);
            classification=z.sign!=0u?1u:(x.sign!=0u||y.sign!=0u?2u:3u);
            if(operation==2u||operation==4u) {
                if(operation==4u&&classification==3u)w.ok=false;
                elMultiply(a[6],x,9u,18u,27u,p,w);elMultiply(a[7],y,9u,18u,27u,q,w);
                elSum(p,q,s,w);elCapacity(s,27u,w);elMultiply(a[8],z,9u,18u,27u,t,w);elSum(s,t,r,w);elCapacity(r,27u,w);sign=r.sign;
            }
            if(operation==5u||operation==6u) {
                elCross(a[6],a[7],a[8],a[9],a[10],a[11],p,q,r,w);
                if(z.sign==0u||r.sign==0u)w.ok=false;
                if(w.ok) {
                    elMultiply(operation==5u?x:y,r,18u,18u,35u,s,w);
                    elMultiply(operation==5u?p:q,z,18u,18u,35u,t,w);
                    if(t.sign!=0u)t.sign=3u-t.sign;elSum(s,t,p,w);sign=p.sign;
                    if(sign!=0u&&z.sign!=r.sign)sign=3u-sign;
                }
            }
        }
    }
    resultWords[4]=w.used;if(!w.ok)return;
    resultWords[0]=1u;resultWords[1]=sign;resultWords[2]=operation==2u?0u:classification;
    if(operation==3u){
        resultWords[5]=x.sign;resultWords[24]=y.sign;resultWords[43]=z.sign;
        for(int k=0;k<18;k++){resultWords[6+k]=x.limb[k];resultWords[25+k]=y.limb[k];resultWords[44+k]=z.limb[k];}
    }
}
)EXACT";
}

inline std::string languageSource(bool metal)
{
    std::string source=metal?
        "#define EL_REF(T,N) thread T& N\n#define EL_VAL(T,N) T N\n#define EL_REQUEST(N) thread const uint* N\n#define EL_RESULT(N) thread uint* N\n":
        "#define EL_REF(T,N) inout T N\n#define EL_VAL(T,N) T N\n#define EL_REQUEST(N) uint N[16]\n#define EL_RESULT(N) out uint N[62]\n";
    source+="uint elBound(uint op){return op==1u?"+std::to_string(maximumVisits(Operation::Det2))+"u:op==2u?"
        +std::to_string(maximumVisits(Operation::Det3))+"u:op==3u?"+std::to_string(maximumVisits(Operation::Cross))
        +"u:op==4u?"+std::to_string(maximumVisits(Operation::Incidence))+"u:op==5u||op==6u?"
        +std::to_string(maximumVisits(Operation::CompareX))+"u:0u;}\n#define EL_BOUND(O) elBound(O)\n";
    source+=arithmeticSource();return source;
}
// Arithmetic fixture entry points. They register no proof/owner/receipt.
inline std::string glslProbeSource()
{
    return "#version 430 core\nlayout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer Inputs {uint inputs[];};\n"
        "layout(std430,binding=1) writeonly buffer Outputs {uint outputs[];};\n"+languageSource(false)+
        "\nvoid main(){uint i=gl_GlobalInvocationID.x;uint a[16],b[62];for(int k=0;k<16;k++)a[k]=inputs[i*16u+uint(k)];"
        "elEvaluate(a,b);for(int k=0;k<62;k++)outputs[i*62u+uint(k)]=b[k];}\n";
}
inline std::string metalProbeSource()
{
    return "#include <metal_stdlib>\nusing namespace metal;\n"+languageSource(true)+
        "\nkernel void exactLineProbe(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],"
        "uint i [[thread_position_in_grid]]){uint a[16],b[62];for(int k=0;k<16;k++)a[k]=inputs[i*16u+uint(k)];"
        "elEvaluate(a,b);for(int k=0;k<62;k++)outputs[i*62u+uint(k)]=b[k];}\n";
}
}
