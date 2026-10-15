#pragma once
#include "fixture_native_camera_wide_line_predicates.h"

namespace arbitgpu::camerapixelsigns
{
// Supplied-bit affine mathematics only. Fixed corners are ordered top-first:
// (0,0),(1,0),(0,1),(1,1). Their values are c,a+c,b+c,a+b+c times the
// canonical form's positive dyadic scale. Zeros, including an all-zero form,
// are retained. No tie, union, owner, visibility or Color classification follows.
// This is an affine certificate, not sampled native visibility or parent capture.
// Throughout the unit square f is the convex combination of these four values
// with weights (1-u)(1-v),u(1-v),(1-u)v,uv. Uniform nonnegative/nonpositive
// corner signs therefore bound the whole affine form. Zero corners stay zero.
constexpr std::uint32_t kVersion=1,kRequestTag=0x43505351u,kWireTag=0x43505352u;
enum class Operation : std::uint32_t { Corners=0 };
constexpr std::uint32_t kRequestWords=65,kResultWords=73,kProbeWords=623;
using Request=std::array<std::uint32_t,kRequestWords>;
using Result=std::array<std::uint32_t,kResultWords>;
using ProbeInput=std::array<std::uint32_t,kProbeWords>;
constexpr std::uint32_t kProbeResultWords=kResultWords+cameraraycoefficients::kResultWords;
using ProbeResult=std::array<std::uint32_t,kProbeResultWords>;
static_assert(sizeof(Request)==260&&sizeof(Result)==292&&sizeof(ProbeResult)==2520,"pixel sign wires");
// Request[0..56] is the exact expected CC parent. Then version/request tag,
// operation=0, full admitted scalar limit, form index0..16, wire tag, two zeros.
// Result status/full REQUIRED scalar charge/wire tag/performed EL visits,
// four signs0/1/2, then exact65-word request echo. Refusal clears status/signs
// and retains charge/tag/echo. Invalid tags/op/index/budget refuse BEFORE parser
// or arithmetic. Caller points and supplied wide integers are not entry points.
// Probe mode0 constructs actual native CC; mode1 is synthetic record math.
// Invalid modes supply a cleared record. Full CC diagnostic follows the signs.

// |a|,|b|,|c|<2^256; each two-term intermediate is <2^257, and the
// three-term corner is <3*2^256<2^258. Nine limbs suffice for all intermediates.
// cwPrepare clears51 ELInt values; three elSum calls each use<=175 visits,
// including clear/magnitude/normalize/capacity. No multiplication/raw-f32 decode.
constexpr std::uint32_t kParsedClearVisits=51*35,kSumCount=3;
constexpr std::uint32_t maximumVisits(){return kParsedClearVisits+kSumCount*175;}
static_assert(maximumVisits()==2310,"pixel sign attempted EL envelope");
// Conservative source word-copy schedule, including GLSL inout copy-in/out:
// wrapper65+557+57+73=752, cpEvaluate557+65+2*73=768;
// cwPrepare557+65+2*(51*36)+2*4=4302, parsed stores51*9=459;
// clear/echo73+65=138, metadata/signs8, ELWork initialization4;
// parser51 elZero calls80 each. Each sum: argument152, zero80, magnitude81,
// normalize80, capacity45; every ELWork step8; two word functions per35-limb
// sum at<=6 moves each. Nested arrays/arguments and refused paths stay charged.
// Probe diagnostic557 stores are a separate557*8 scalar charge below.
constexpr std::uint64_t maximumWordMoves()
{return 752+768+4302+459+138+8+4+51*80+3*(152+80+81+80+45)+8*maximumVisits()+6*70*3;}
// cwPrepare has57 echo+19 form+53 scalar+424 magnitude iterations=553,
// <=48 elementary operations each, plus<=2048 parser/setup/refusal operations,
// rounded to32768. Remaining wire/index loops use<=8 per moved word. EL scalar
// operations amortize to<=64 per reserved visit over each complete175-visit sum
// and each35-visit clear; an individual signed-sum iteration can exceed64.
// FULL charge precedes every refusal.
// These prospective source envelopes are not GPU instruction/spill measurements.
constexpr std::uint32_t maximumScalarOperations()
{return static_cast<std::uint32_t>(32768+64ull*maximumVisits()+8*maximumWordMoves());}
static_assert(maximumWordMoves()==31565&&maximumScalarOperations()==433128,"complete pixel sign scalar envelope");
constexpr std::uint64_t kProbeDiagnosticScalarOperations=557*8;
// Logical live bound: four parsed51-integer arrays,32 nested EL integer copies,
// four CC records, three requests/results and1024 scalar bytes. The32 integers
// cover three sum outputs and deepest sum/magnitude/clear/normalize/capacity
// argument copies with room for all three result locals. No point arrays exist.
// Actual compiled pipelines/spills/buffers and retained owners remain separate.
constexpr std::uint64_t kLogicalScratchBytes=4*51*144+32*144+4*557*4+3*65*4+3*73*4+1024;
static_assert(kLogicalScratchBytes==45576,"logical sign scratch excluding constructor");
// Host preparation uses the protected canonical record validator,57-word parent
// initialization plus57-word copy and<=128 outer gate operations. Payload is record+request;
// constructor/optional conversion can copy it twice. Later caller copies,
// retention and native buffers are NOT capped by this mathematical API.
constexpr std::uint64_t kHostPrepareWordMoves=2*57;
constexpr std::uint64_t kHostValidationScalarOperations=camerawideline::kHostValidationScalarOperations+8*kHostPrepareWordMoves+128;
constexpr std::uint64_t kPreparedPayloadBytes=(557+65)*4,kFactoryCopyBytes=2*kPreparedPayloadBytes;
inline bool validRequest(const Request& q)
{return q[57]==kVersion&&q[58]==kRequestTag&&q[59]==0&&q[60]>=maximumScalarOperations()
    &&q[61]<17&&q[62]==kWireTag&&!q[63]&&!q[64];}
class PreparedForm
{
public:
    static std::optional<PreparedForm> prepare(const cameraraycoefficients::Result& record,const Request& request)
    {
        if(!validRequest(request))return std::nullopt;
        cameraraycoefficients::Request parent{};for(unsigned k=0;k<57;k++)parent[k]=request[k];
        if(!camerawideline::validateRecord(record,parent))return std::nullopt;
        return PreparedForm(record,request);
    }
    const cameraraycoefficients::Result& record()const{return record_;}
    const Request& request()const{return request_;}
private:
    PreparedForm(const cameraraycoefficients::Result& record,const Request& request):record_(record),request_(request){}
    const cameraraycoefficients::Result record_;
    const Request request_;
};
inline const char* arithmeticSource()
{
    return R"PIXELSIGNS(
void cpEvaluate(CP_RECORD(recordWords),CP_REQUEST(requestWords),CP_RESULT(resultWords)){
    for(int k=0;k<73;k++)resultWords[k]=0u;
    for(int k=0;k<65;k++)resultWords[8+k]=requestWords[k];
    resultWords[1]=CP_CHARGE;resultWords[2]=CP_WIRE;
    if(requestWords[57]!=1u||requestWords[58]!=CP_REQUEST_TAG||requestWords[59]!=0u
        ||requestWords[60]<CP_CHARGE||requestWords[61]>=17u||requestWords[62]!=CP_WIRE
        ||requestWords[63]!=0u||requestWords[64]!=0u)return;
    ELWork w;w.used=0u;w.limit=CP_VISITS;w.capacity=9u;w.ok=true;
    CWLine lines[17];if(!cwPrepare(recordWords,requestWords,lines,w)){resultWords[3]=w.used;return;}
    uint form=requestWords[61];ELInt ac,bc,abc;
    elSum(lines[form].a[0],lines[form].a[2],ac,w);
    elSum(lines[form].a[1],lines[form].a[2],bc,w);
    elSum(ac,lines[form].a[1],abc,w);
    resultWords[3]=w.used;if(!w.ok)return;
    resultWords[0]=1u;resultWords[4]=lines[form].a[2].sign;
    resultWords[5]=ac.sign;resultWords[6]=bc.sign;resultWords[7]=abc.sign;
}
)PIXELSIGNS";
}
inline std::string languageSource(bool metal)
{
    // Reuse the protected canonical parser and low-level EL sums only.
    // elEvaluate/elDecode are never called by this entry. CC probe work is separate.
    auto source=camerawideline::languageSource(metal);
    source+=metal?"\n#define CP_RECORD(N) thread const uint (&N)[557]\n#define CP_REQUEST(N) thread const uint (&N)[65]\n#define CP_RESULT(N) thread uint (&N)[73]\n":
        "\n#define CP_RECORD(N) uint N[557]\n#define CP_REQUEST(N) uint N[65]\n#define CP_RESULT(N) inout uint N[73]\n";
    source+="#define CP_CHARGE "+std::to_string(maximumScalarOperations())+"u\n#define CP_VISITS "+std::to_string(maximumVisits())+
        "u\n#define CP_REQUEST_TAG "+std::to_string(kRequestTag)+"u\n#define CP_WIRE "+std::to_string(kWireTag)+"u\n";
    return source+arithmeticSource();
}
inline std::string probeBody()
{
    return "uint a[65],r[557],b[73];for(int k=0;k<65;k++)a[k]=inputs[i*623u+uint(k)];"
        "uint mode=inputs[i*623u+65u];if(mode==0u){uint c[57];for(int k=0;k<57;k++)c[k]=a[k];ccEvaluate(c,r);}"
        "else{for(int k=0;k<557;k++)r[k]=mode==1u?inputs[i*623u+66u+uint(k)]:0u;}"
        "cpEvaluate(r,a,b);for(int k=0;k<73;k++)outputs[i*630u+uint(k)]=b[k];"
        "for(int k=0;k<557;k++)outputs[i*630u+73u+uint(k)]=r[k];";
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
        "\nkernel void cameraPixelSignProbe(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],"
        "uint i [[thread_position_in_grid]]){"+probeBody()+"}\n";
}
} // namespace arbitgpu::camerapixelsigns
