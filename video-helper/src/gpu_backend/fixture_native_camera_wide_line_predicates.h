#pragma once
#include "fixture_native_camera_line_predicates.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace arbitgpu::camerawideline
{
// Pure generated-integer mathematics, never capture/owner/visibility/coverage
// authority. Query intervals, raster association and radial certificates remain
// unbound. Precision30 API/charges and the raw-f32 decoder remain unchanged.
// Actual queries use object-local vertices with rounded inverse transforms;
// raster attribution uses separately transformed world triangles. Supplied bits
// and shared IDs do not prove those geometries equivalent. Exact equal events
// here do not substitute for released float-distance/table/instance/triangle ties.
using Operation=cameraline::Operation;
using Classification=cameraline::Classification;
using Request=cameraline::Request;
using ProbeInput=cameraline::ProbeInput;
constexpr std::uint32_t kVersion=1,kRequestWords=65,kProbeWords=623,kResultWords=177;
using Result=std::array<std::uint32_t,kResultWords>;
constexpr std::uint32_t kProbeResultWords=kResultWords+cameraraycoefficients::kResultWords;
using ProbeResult=std::array<std::uint32_t,kProbeResultWords>;
constexpr std::uint64_t kProbeDiagnosticScalarOperations=557*8;
static_assert(sizeof(Result)==708,"wide generated predicate wire");
static_assert(sizeof(ProbeResult)==2936,"constructor diagnostic probe wire");
// The 65-word closed request has the precision30 operation layout, but its exact
// expected constructor request selects precision256. No supplied point wire.
// Result[0..3] = supported/full REQUIRED scalar charge/classification/sign;
// [4..111] = two points, three [sign,17 low limbs] each; [112..176] request echo.
// Finite points normalize z>0. Coincidence requires two nonzero forms; a zero
// form has its own classification. Equal incidence/order returns zero, no tie.
// Failure clears mathematics only, retaining full charge and exact request.
// Probe mode0 runs the actual native constructor; mode1 is explicitly synthetic
// arithmetic input, not constructor evidence or a native parent/proof receipt.
// Probe readback appends the full constructor record after177 predicate words,
// allowing independent verification of actual precision30 constructor refusal.
// These557 additional diagnostic stores are charged separately from the kernel.
// Eight-word magnitudes are <2^256. det2 <2^513, three-product incidence
// <3*2^769<2^771, coordinate-product difference <2^1027. Intermediates, not just
// final signs, fit17/25/33 limbs. Common positive dyadic scales cancel only after
// canonical validation. No generated coefficient is decoded/rounded to f32.

constexpr std::uint32_t kParsedClearVisits=51*35,kPointClearVisits=6*35;
// elCross uses three elDet2 calls, each with two9x9 products, elSum and an
// extra35-word cap scan. Then this layer scans all three points against17 limbs.
constexpr std::uint32_t kCrossVisits=exactline::kCrossVisits+3*35;
constexpr std::uint32_t kIncidenceVisits=3*exactline::multiplyVisits(8,17)+2*175+35;
constexpr std::uint32_t kCompareVisits=2*exactline::multiplyVisits(17,17)+175+35;
constexpr std::uint32_t crossCount(Operation op){return op==Operation::Cross||op==Operation::Incidence?1:2;}
constexpr std::uint32_t incidenceCount(Operation op){return op==Operation::Incidence?1:0;}
constexpr std::uint32_t comparisonCount(Operation op){return op==Operation::Cross||op==Operation::Incidence?0:1;}
constexpr std::uint32_t maximumVisits(Operation op)
{return kParsedClearVisits+kPointClearVisits+crossCount(op)*kCrossVisits+
    incidenceCount(op)*kIncidenceVisits+comparisonCount(op)*kCompareVisits;}
static_assert(kCrossVisits==36291&&kIncidenceVisits==29878&&kCompareVisits==41598,"wide arithmetic schedules");
static_assert(arbitgpu::camerawideline::maximumVisits(Operation::Cross)==38286&&arbitgpu::camerawideline::maximumVisits(Operation::Incidence)==68164
    &&arbitgpu::camerawideline::maximumVisits(Operation::CompareX)==116175,"complete attempted visit envelopes");
// Explicit worst copy schedule, including GLSL inout copy-in/out even when a
// compiler can optimize it away; Metal assignments are covered by the same bound.
// ELInt=36 words, ELWork=4. The fixed prefix rounds8028 moves to8192:
// wrapper799, cwEvaluate arguments976, cwPrepare arguments4294, parsed stores459,
// result clear/echo242, point output114, provenance8, two cwCross calls892,
// max(cwIncidence226,cwCompare229), result metadata4, local ELWork fields4,
// denominator sign normalization6 and comparison sign negation1.
// Diagnostic557 stores and constructor copies are separately charged. Every
// word move includes its loop/index costs below. Internal EL copies are separate.
constexpr std::uint64_t maximumWordMoves(Operation op)
{
    const std::uint64_t c=crossCount(op),i=incidenceCount(op),v=comparisonCount(op);
    const auto products=486*c+408*i+578*v,multiplies=6*c+3*i+2*v,sums=3*c+2*i+v;
    // elCross440; elDet2224; multiply155; sum152; zero80; capacity45;
    // normalize80; magnitude80; accumulate82; step8; product<=8; wordfunc<=6.
    // 2 full35-step carry walks/product, 2 wordfuncs/35-step signed sum.
    return 8192+440*c+224*3*c+155*multiplies+152*sums+
        80*(51+6+multiplies+sums)+45*(3*multiplies+sums+6*c+i+v)+
        80*(multiplies+sums)+80*sums+82*2*products+8*arbitgpu::camerawideline::maximumVisits(op)+
        8*products+6*(70*products+70*sums);
}
// Native parser has57 echo+19 form+53 scalar+424 magnitude iterations=553,
// <=48 scalar operations each, including scalar sign/canonical branches, nested
// array indices and loop setup; EL clears/copies are separate. <=2048 remaining
// setup/sign/return operations. 28592 rounded to32768. Wire/copy loops including
// indices cost<=8 per moved word.
// Reserve64 per worst-case EL visit for arithmetic/index work and amortized
// setup, separately from copies. A sum's35 arithmetic iterations can exceed64;
// <=100 each plus140 clear/scan iterations<=32 totals7980 <=64*175=11200.
// The complete reserved schedule covers failed loops too.
// ELWork saturation is NOT a physical stop; FULL admission precedes parsing and
// every attempted dispatch reports the full envelope even on refusal.
// These are prospective elementary source-operation/move counts, not measured
// GPU instructions or a charge for actual compiled spill/pipeline resources.
constexpr std::uint32_t maximumScalarOperations(Operation op)
{return static_cast<std::uint32_t>(32768+64ull*arbitgpu::camerawideline::maximumVisits(op)+8*maximumWordMoves(op));}
static_assert(maximumWordMoves(Operation::CompareX)==1874860,"nested comparison copies");
static_assert(arbitgpu::camerawideline::maximumScalarOperations(Operation::Cross)==7390312
    &&arbitgpu::camerawideline::maximumScalarOperations(Operation::Incidence)==13171856
    &&arbitgpu::camerawideline::maximumScalarOperations(Operation::CompareX)==22466848,"native scalar envelopes");
// Host validation is separate: at most553 bounded iterations*48+2048 setup,
// no native arithmetic. Prepared payload=2456 bytes. Factory construction and
// optional conversion can copy that payload twice; later caller copies/retention
// are NOT capped by this API and must be admitted by the whole native owner.
constexpr std::uint64_t kHostValidationScalarOperations=553*48+2048;
constexpr std::uint64_t kPreparedPayloadBytes=(557+57)*4,kFactoryCopyBytes=2*kPreparedPayloadBytes;
// Native logical live bound:4 prepared arrays(51*144),96 nested integer/value
// copies,8 point/provenance records(440),4 records,3 requests,3 results,1024 scalar
// bytes. Not compiled spills/pipeline, buffers, host prepared/caller or incumbent
// resources. Constructor scratch/work are additional, not included here.
constexpr std::uint64_t kLogicalScratchBytes=4*51*144+96*144+8*440+4*557*4+3*65*4+3*177*4+1024;
static_assert(kLogicalScratchBytes==59560,"wide logical native scratch");
// No scalar-operation to released geometry-query-work conversion is asserted.
// Constructor227334496 source-cost units is PER two-triangle request, not ALL-T
// scene admission. Full arrangement/resource admission remains a later task.

inline bool canonical(const std::uint32_t* values,unsigned count,std::uint32_t exponent)
{
    if(exponent>512u&&exponent<0xfffffe00u)return false;
    std::uint32_t odd=0,any=0;
    for(unsigned j=0;j<count;j++){
        const auto* a=values+j*9;std::uint32_t magnitude=0;
        for(unsigned k=1;k<9;k++)magnitude|=a[k];
        if(a[0]>2||(magnitude==0)!=(a[0]==0))return false;
        odd|=a[1]&1u;any|=magnitude;
    }
    return any?odd!=0:exponent==0;
}
inline bool validateRecord(const cameraraycoefficients::Result& record,const cameraraycoefficients::Request& expected)
{
    namespace cc=cameraraycoefficients;
    if(expected[0]!=cc::kVersion||expected[1]!=256||expected[2]<cc::maximumVisits(256)||expected[3]
        ||record[0]!=1||record[1]!=cc::maximumVisits(256)||record[2]==0||record[2]>record[1]||record[3]!=256)return false;
    for(unsigned k=0;k<57;k++)if(record[4+k]!=expected[k])return false;
    for(unsigned k=0;k<17;k++)if(!canonical(record.data()+62+28*k,3,record[61+28*k]))return false;
    return record[538]!=0&&record[548]!=0&&canonical(record.data()+538,1,record[537])&&canonical(record.data()+548,1,record[547]);
}
// Immutable shape/echo data. A fabricated mathematical record can pass shape
// validation; neither this object nor a shader status authenticates native inputs.
class PreparedRecord
{
public:
    static std::optional<PreparedRecord> prepare(const cameraraycoefficients::Result& r,const cameraraycoefficients::Request& q)
    {
        if(!validateRecord(r,q))return std::nullopt;
        return PreparedRecord(r,q);
    }
    const cameraraycoefficients::Result& record()const{return record_;}
    const cameraraycoefficients::Request& request()const{return request_;}
private:
    PreparedRecord(const cameraraycoefficients::Result& r,const cameraraycoefficients::Request& q):record_(r),request_(q){}
    const cameraraycoefficients::Result record_;
    const cameraraycoefficients::Request request_;
};

inline const char* arithmeticSource()
{
return R"CAMERAWIDE(
struct CWLine { ELInt a[3]; };
struct CWPoint { ELInt a[3]; uint first; uint second; };
// Internal locals only; the closed cwEvaluate entry has no wide-point input.
bool cwPrepare(CW_RECORD(recordWords),CW_REQUEST(requestWords),CW_LINES(lines),EL_REF(ELWork,w)){
    if(requestWords[0]!=1u||requestWords[1]!=256u||requestWords[2]<1208219u||requestWords[3]!=0u
        ||recordWords[0]!=1u||recordWords[1]!=1208219u||recordWords[2]==0u||recordWords[2]>1208219u||recordWords[3]!=256u)return false;
    for(int k=0;k<57;k++)if(recordWords[4+k]!=requestWords[k])return false;
    for(int k=0;k<19;k++){
        int at=k<17?61+28*k:k==17?537:547,count=k<17?3:1;
        uint exponent=recordWords[at];if(exponent>512u&&exponent<4294966784u)return false;
        uint odd=0u,any=0u;
        for(int j=0;j<count;j++){
            int p=at+1+9*j;uint sign=recordWords[p],magnitude=0u;
            if(sign>2u)return false;
            if(k<17)elZero(lines[k].a[j],w);
            for(int n=0;n<8;n++){
                uint word=recordWords[p+1+n];magnitude|=word;
                if(k<17)lines[k].a[j].limb[n]=word;
            }
            if((magnitude==0u)!=(sign==0u))return false;
            odd|=recordWords[p+1]&1u;any|=magnitude;
            if(k<17)lines[k].a[j].sign=sign;
        }
        if(any!=0u?odd==0u:exponent!=0u)return false;
        if(k>=17&&any==0u)return false;
    }
    return w.ok;
}
uint cwCross(EL_VAL(CWLine,a),EL_VAL(CWLine,b),uint first,uint second,
    EL_REF(CWPoint,p),EL_REF(ELWork,w)){
    p.first=first;p.second=second;
    elCross(a.a[0],a.a[1],a.a[2],b.a[0],b.a[1],b.a[2],p.a[0],p.a[1],p.a[2],w);
    for(int k=0;k<3;k++)elCapacity(p.a[k],17u,w);
    bool az=a.a[0].sign==0u&&a.a[1].sign==0u&&a.a[2].sign==0u;
    bool bz=b.a[0].sign==0u&&b.a[1].sign==0u&&b.a[2].sign==0u;
    if(az||bz)return 4u;
    if(p.a[2].sign!=0u){if(p.a[2].sign==2u)for(int k=0;k<3;k++)if(p.a[k].sign!=0u)p.a[k].sign=3u-p.a[k].sign;return 1u;}
    return p.a[0].sign!=0u||p.a[1].sign!=0u?2u:3u;
}
uint cwIncidence(EL_VAL(CWLine,line),EL_VAL(CWPoint,p),EL_REF(ELWork,w)){
    ELInt t[3],s,r;for(int k=0;k<3;k++)elMultiply(line.a[k],p.a[k],8u,17u,25u,t[k],w);
    elSum(t[0],t[1],s,w);elSum(s,t[2],r,w);elCapacity(r,25u,w);return r.sign;
}
uint cwCompare(EL_VAL(CWPoint,p),EL_VAL(CWPoint,q),int axis,EL_REF(ELWork,w)){
    if(p.a[2].sign!=1u||q.a[2].sign!=1u){w.ok=false;return 0u;}
    ELInt x,y,r;elMultiply(p.a[axis],q.a[2],17u,17u,33u,x,w);elMultiply(q.a[axis],p.a[2],17u,17u,33u,y,w);
    if(y.sign!=0u)y.sign=3u-y.sign;elSum(x,y,r,w);elCapacity(r,33u,w);return r.sign;
}
void cwEvaluate(CW_RECORD(recordWords),CW_REQUEST(requestWords),CW_RESULT(resultWords)){
    for(int k=0;k<177;k++)resultWords[k]=0u;
    for(int k=0;k<65;k++)resultWords[112+k]=requestWords[k];
    uint op=requestWords[58],charge=CW_CHARGE(op);resultWords[1]=charge;
    if(requestWords[57]!=1u||op>3u||requestWords[59]<charge||requestWords[64]!=0u)return;
    uint a=requestWords[60],b=requestWords[61],c=requestWords[62],d=requestWords[63];
    if(a>=17u||b>=17u||c>=17u||d>=17u||(op==0u&&(c!=0u||d!=0u))||(op==1u&&d!=0u))return;
    ELWork w;w.used=0u;w.limit=CW_VISITS(op);w.capacity=35u;w.ok=true;
    CWLine lines[17];if(!cwPrepare(recordWords,requestWords,lines,w))return;
    CWPoint p,q;p.first=0u;p.second=0u;q.first=0u;q.second=0u;
    for(int k=0;k<3;k++){elZero(p.a[k],w);elZero(q.a[k],w);}
    uint category=cwCross(lines[a],lines[b],a,b,p,w),sign=0u;
    if(op==1u){if(category==3u||category==4u)return;sign=cwIncidence(lines[c],p,w);}
    if(op>=2u){uint second=cwCross(lines[c],lines[d],c,d,q,w);if(category!=1u||second!=1u)return;
        sign=cwCompare(p,q,op==2u?0:1,w);}
    if(!w.ok)return;
    resultWords[0]=1u;resultWords[2]=category;resultWords[3]=sign;
    for(int k=0;k<3;k++){
        resultWords[4+18*k]=p.a[k].sign;resultWords[58+18*k]=q.a[k].sign;
        for(int j=0;j<17;j++){resultWords[5+18*k+j]=p.a[k].limb[j];resultWords[59+18*k+j]=q.a[k].limb[j];}
    }
}
)CAMERAWIDE";
}
inline std::string languageSource(bool metal)
{
    // No call to elEvaluate/elDecode. CC body is used only by the probe wrapper.
    auto source=cameraraycoefficients::languageSource(metal);
    source+=metal?"\n#define CW_RECORD(N) thread const uint (&N)[557]\n#define CW_REQUEST(N) thread const uint (&N)[65]\n#define CW_RESULT(N) thread uint (&N)[177]\n#define CW_LINES(N) thread CWLine (&N)[17]\n":
        "\n#define CW_RECORD(N) uint N[557]\n#define CW_REQUEST(N) uint N[65]\n#define CW_RESULT(N) inout uint N[177]\n#define CW_LINES(N) inout CWLine N[17]\n";
    source+="uint cwCharge(uint op){return op==0u?"+std::to_string(arbitgpu::camerawideline::maximumScalarOperations(Operation::Cross))+"u:op==1u?"+
        std::to_string(arbitgpu::camerawideline::maximumScalarOperations(Operation::Incidence))+"u:"+std::to_string(arbitgpu::camerawideline::maximumScalarOperations(Operation::CompareX))+"u;}\n";
    source+="uint cwVisits(uint op){return op==0u?"+std::to_string(arbitgpu::camerawideline::maximumVisits(Operation::Cross))+"u:op==1u?"+
        std::to_string(arbitgpu::camerawideline::maximumVisits(Operation::Incidence))+"u:"+std::to_string(arbitgpu::camerawideline::maximumVisits(Operation::CompareX))+"u;}\n#define CW_CHARGE(O) cwCharge(O)\n#define CW_VISITS(O) cwVisits(O)\n";
    return source+arithmeticSource();
}
inline std::string probeBody()
{
    return "uint a[65],r[557],b[177];for(int k=0;k<65;k++)a[k]=inputs[i*623u+uint(k)];"
        "uint mode=inputs[i*623u+65u];if(mode==0u){uint c[57];for(int k=0;k<57;k++)c[k]=a[k];ccEvaluate(c,r);}"
        "else{for(int k=0;k<557;k++)r[k]=mode==1u?inputs[i*623u+66u+uint(k)]:0u;}"
        "cwEvaluate(r,a,b);for(int k=0;k<177;k++)outputs[i*734u+uint(k)]=b[k];"
        "for(int k=0;k<557;k++)outputs[i*734u+177u+uint(k)]=r[k];";
}
inline std::string glslProbeSource()
{
    return "#version 430 core\nlayout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer Inputs{uint inputs[];};\n"
        "layout(std430,binding=1) writeonly buffer Outputs{uint outputs[];};\n"+languageSource(false)+
        "\nvoid main(){uint i=gl_GlobalInvocationID.x;"+probeBody()+"}\n";
}
inline std::string metalProbeSource()
{
    return "#include <metal_stdlib>\nusing namespace metal;\n"+languageSource(true)+
        "\nkernel void cameraWideLinePredicateProbe(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],"
        "uint i [[thread_position_in_grid]]){"+probeBody()+"}\n";
}
} // namespace arbitgpu::camerawideline
