#pragma once
#include "fixture_native_camera_ray_coefficients.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>

namespace arbitgpu::cameraline
{
// Mathematical data only. Neither constructor status, echoed tags nor these
// integers authenticate captured uniforms, a native owner, visibility or coverage.
// Camera query intervals and whole-cell radial certificates remain unbound.
enum class Operation : std::uint32_t { Cross=0, Incidence=1, CompareX=2, CompareY=3 };
enum class Classification : std::uint32_t { Finite=1, Parallel=2, Coincident=3, ZeroLine=4 };
constexpr std::uint32_t kVersion=1,kRequestWords=65,kResultWords=99,kProbeWords=623;
using Request=std::array<std::uint32_t,kRequestWords>;
using Result=std::array<std::uint32_t,kResultWords>;
using ProbeInput=std::array<std::uint32_t,kProbeWords>;
static_assert(sizeof(Request)==260&&sizeof(Result)==396&&sizeof(ProbeInput)==2492,"generated-line wire layout");
// For |coefficient|<2^30: two products differ by <2^61; three line*point
// products sum to <3*2^91<2^93; two point-coordinate products differ by <2^123.
// Four 32-bit limbs cover every intermediate. Denominator sign normalization
// changes signs only. This is NOT an arbitrary 32-bit or 256-bit input domain.
// Request[0..56]: exact expected constructor request. Then version, operation,
// FULL admitted scalar-operation limit, a,b,c,d form indices, reserved zero.
// Cross(a,b); incidence(c,cross(a,b)); compare cross(a,b) with cross(c,d).
// Unused indices must be zero. No input wire accepts wide point coordinates.
// Result: supported, full REQUIRED charge (also on failure), classification of
// first point, sign (0/1/2). Two points, each three [sign,4 little-endian limbs]
// occupy [4..33], followed by exact request echo [34..98]. Points at infinity
// retain orientation; finite points are multiplied by -1 if needed so z>0.
// Incidence refuses a coincident/zero intersection; comparison requires finite
// points. Cross reports zero-line, parallel and coincident without inventing ties.
// All mathematical output is zero on refusal; the full charge and echo survive.
// Probe wire: request[65], mode (0=native constructor,1=synthetic arithmetic
// fixture ONLY), constructor record[557]. Mode1 is never constructor evidence.

constexpr std::uint32_t multiplyVisits(std::uint32_t m,std::uint32_t n){return 12+9*m*n;}
constexpr std::uint32_t kSumVisits=17;
constexpr std::uint32_t kCrossVisits=6*multiplyVisits(1,1)+3*kSumVisits;
constexpr std::uint32_t kIncidenceVisits=3*multiplyVisits(1,2)+2*kSumVisits;
constexpr std::uint32_t kCompareVisits=2*multiplyVisits(2,2)+kSumVisits;
constexpr std::uint32_t maximumVisits(Operation op)
{return op==Operation::Cross?kCrossVisits:op==Operation::Incidence?kCrossVisits+kIncidenceVisits:
    2*kCrossVisits+kCompareVisits;}
// Disjoint parser/setup/wire charges: clPrepare has 57 echo comparisons +19
// form iterations +53 scalar iterations +371 upper-limb iterations =500, each
// <=24 scalar operations. Its 204 zero stores belong to wire moves, not parser.
// <=8192 word moves at <=8 operations INCLUDING loop/index costs: wrapper wire,
// result clearing/echo/stores, point/prepared clears, and nested GLSL parameter
// copy-in/out/Metal assignments (including record and prepared-array copies).
// includes clAccumulate/clZero/clNormalize copy-in/out, not just outer calls.
// Worst comparison copies: <2700 outer wire/array words, <600 prepared/result
// stores, <850 cross/det/multiply/sum parameter words, <600 accumulate parameter
// words, <1600 clear/normalize/negate parameter words; total <6350<8192.
// <=1024 remaining setup/sign/classification/return operations. 78560 rounded
// upward to 81920. Arithmetic clears/scans are charged ONLY as visits below.
// Arithmetic: each visit <=64 scalar operations INCLUDING split16 word helpers.
// No saturating counter pretends to halt loops. The FULL envelope is admitted
// before parsing, and charged on every failure. Constructor work is SEPARATE.
constexpr std::uint32_t maximumScalarOperations(Operation op)
{return 81920+64*maximumVisits(op);}
static_assert(kCrossVisits==177&&kIncidenceVisits==124&&kCompareVisits==113,"four-limb schedules");
// Logical live upper bound: 4 prepared arrays (17*3 signed 20-byte integers),
// 96 arithmetic/line/point/value-parameter integers, 8 point/provenance records,
// four constructor records, two requests, three outputs and 1024 scalar bytes.
// This includes nested helper/value copies, not compiler spills, pipeline,
// native/retained/caller buffers or constructor scratch, admitted separately.
constexpr std::uint64_t kLogicalScratchBytes=4*17*3*20+96*20+8*68+4*557*4+2*65*4+3*99*4+1024;
static_assert(kLogicalScratchBytes==18188,"logical four-limb predicate scratch");

inline bool canonical(const std::uint32_t* values,unsigned count,std::uint32_t exponent)
{
    // Twos-complement wire exponent [-512,512], without out-of-range int casts.
    if(exponent>512u&&exponent<0xfffffe00u)return false;
    std::uint32_t odd=0,nonzero=0;
    for(unsigned j=0;j<count;j++){
        const auto* a=values+j*9;const auto sign=a[0],word=a[1];
        if(sign>2||word>=0x40000000u||(word==0)!=(sign==0))return false;
        for(unsigned k=2;k<9;k++)if(a[k])return false;
        odd|=word&1u;nonzero|=word;
    }
    return nonzero?odd!=0:exponent==0;
}
inline bool validateRecord(const cameraraycoefficients::Result& record,
    const cameraraycoefficients::Request& expected)
{
    namespace cc=cameraraycoefficients;
    if(expected[0]!=cc::kVersion||expected[1]!=30||expected[2]<cc::maximumVisits(30)||expected[3]
        ||record[0]!=1||record[1]!=cc::maximumVisits(30)||record[2]==0||record[2]>record[1]||record[3]!=30)return false;
    for(unsigned k=0;k<cc::kRequestWords;k++)if(record[4+k]!=expected[k])return false;
    for(unsigned k=0;k<cc::kLineCount;k++)if(!canonical(record.data()+62+28*k,3,record[61+28*k]))return false;
    return record[538]!=0&&record[548]!=0&&canonical(record.data()+538,1,record[537])&&canonical(record.data()+548,1,record[547]);
}
// Immutable validated shape + exact request/opaque endpoint/order tags. A caller
// can fabricate mathematical data; this class is explicitly NOT native proof.
class PreparedRecord
{
public:
    static std::optional<PreparedRecord> prepare(const cameraraycoefficients::Result& record,
        const cameraraycoefficients::Request& expected)
    {
        if(!validateRecord(record,expected))return std::nullopt;
        return PreparedRecord(record,expected);
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
return R"CAMERALINE(
struct CLInt { uint limb[4]; uint sign; };
struct CLLine { CLInt a[3]; };
struct CLPoint { CLInt a[3]; uint first; uint second; };
// Internal routines consume locals made in the closed clEvaluate operation.
// They are not an external point input API or an authentication mechanism.
void clZero(EL_REF(CLInt,r)){for(int k=0;k<4;k++)r.limb[k]=0u;r.sign=0u;}
void clNormalize(EL_REF(CLInt,r)){
    uint any=0u;for(int k=0;k<4;k++)any|=r.limb[k];if(any==0u)r.sign=0u;
}
void clAccumulate(EL_REF(CLInt,r),int at,uint value,EL_REF(bool,ok)){
    for(int k=0;k<4;k++){
        int index=at+k;uint carry=0u;
        if(index<4){r.limb[index]=elAddWord(r.limb[index],value,carry);value=carry;}
        else if(value!=0u){ok=false;value=0u;}
    }
    if(value!=0u)ok=false;
}
// Inputs have statically known active widths; upper limbs are checked. Each
// product: 4 clear +4 upper-width validation +4 zero scan +9*m*n visits.
void clMultiply(EL_VAL(CLInt,a),EL_VAL(CLInt,b),int m,int n,EL_REF(CLInt,r),EL_REF(bool,ok)){
    clZero(r);for(int k=0;k<4;k++)if((k>=m&&a.limb[k]!=0u)||(k>=n&&b.limb[k]!=0u))ok=false;
    for(int i=0;i<m;i++)for(int j=0;j<n;j++){
        uint hi=0u,lo=0u;elProduct(a.limb[i],b.limb[j],hi,lo);
        clAccumulate(r,i+j,lo,ok);clAccumulate(r,i+j+1,hi,ok);
    }
    r.sign=a.sign==0u||b.sign==0u?0u:a.sign==b.sign?1u:2u;clNormalize(r);
}
void clSum(EL_VAL(CLInt,a),EL_VAL(CLInt,b),EL_REF(CLInt,r),EL_REF(bool,ok)){
    clZero(r);int cmp=0;
    for(int k=3;k>=0;k--)if(cmp==0&&a.limb[k]!=b.limb[k])cmp=a.limb[k]>b.limb[k]?1:-1;
    bool add=a.sign==b.sign,first=cmp>=0;uint carry=0u;
    for(int k=0;k<4;k++){
        uint x=first?a.limb[k]:b.limb[k],y=first?b.limb[k]:a.limb[k],next=0u,extra=0u;
        uint part=add?elAddWord(x,y,next):elSubWord(x,y,next);
        r.limb[k]=add?elAddWord(part,carry,extra):elSubWord(part,carry,extra);carry=next|extra;
    }
    if(carry!=0u)ok=false;r.sign=add?a.sign:first?a.sign:b.sign;clNormalize(r);
}
void clNegate(EL_REF(CLInt,a)){if(a.sign!=0u)a.sign=3u-a.sign;}
void clDet(EL_VAL(CLInt,a),EL_VAL(CLInt,b),EL_VAL(CLInt,c),EL_VAL(CLInt,d),int m,int n,
    EL_REF(CLInt,r),EL_REF(bool,ok)){
    CLInt x,y;clMultiply(a,d,m,n,x,ok);clMultiply(b,c,m,n,y,ok);clNegate(y);clSum(x,y,r,ok);
}
// A point only originates here, from validated 30-bit lines. No wide point
// parser exists. Provenance indices remain attached even on equal intersections.
uint clCross(EL_VAL(CLLine,a),EL_VAL(CLLine,b),uint first,uint second,EL_REF(CLPoint,p),EL_REF(bool,ok)){
    p.first=first;p.second=second;
    for(int k=0;k<3;k++)clDet(a.a[(k+1)%3],a.a[(k+2)%3],b.a[(k+1)%3],b.a[(k+2)%3],1,1,p.a[k],ok);
    bool az=a.a[0].sign==0u&&a.a[1].sign==0u&&a.a[2].sign==0u;
    bool bz=b.a[0].sign==0u&&b.a[1].sign==0u&&b.a[2].sign==0u;
    if(az||bz)return 4u;
    if(p.a[2].sign!=0u){if(p.a[2].sign==2u)for(int k=0;k<3;k++)clNegate(p.a[k]);return 1u;}
    return p.a[0].sign!=0u||p.a[1].sign!=0u?2u:3u;
}
uint clIncidence(EL_VAL(CLLine,line),EL_VAL(CLPoint,p),EL_REF(bool,ok)){
    CLInt t[3],s,r;for(int k=0;k<3;k++)clMultiply(line.a[k],p.a[k],1,2,t[k],ok);
    clSum(t[0],t[1],s,ok);clSum(s,t[2],r,ok);return r.sign;
}
uint clCompare(EL_VAL(CLPoint,p),EL_VAL(CLPoint,q),int axis,EL_REF(bool,ok)){
    if(p.a[2].sign!=1u||q.a[2].sign!=1u){ok=false;return 0u;}
    CLInt r;clDet(p.a[axis],p.a[2],q.a[axis],q.a[2],2,2,r,ok);return r.sign;
}
bool clPrepare(CL_RECORD(recordWords),CL_REQUEST(requestWords),CL_LINES(lines)){
    if(requestWords[0]!=1u||requestWords[1]!=30u||requestWords[2]<210740u||requestWords[3]!=0u
        ||recordWords[0]!=1u||recordWords[1]!=210740u||recordWords[2]==0u||recordWords[2]>210740u||recordWords[3]!=30u)return false;
    for(int k=0;k<57;k++)if(recordWords[4+k]!=requestWords[k])return false;
    // Inline validation avoids a full 557-word GLSL parameter copy per form.
    for(int k=0;k<19;k++){
        int at=k<17?61+28*k:k==17?537:547,count=k<17?3:1;
        uint exponent=recordWords[at];if(exponent>512u&&exponent<4294966784u)return false;
        uint odd=0u,any=0u;
        for(int j=0;j<count;j++){
            int p=at+1+j*9;uint sign=recordWords[p],word=recordWords[p+1];
            if(sign>2u||word>=1073741824u||((word==0u)!=(sign==0u)))return false;
            for(int n=2;n<9;n++)if(recordWords[p+n]!=0u)return false;
            odd|=word&1u;any|=word;
            if(k<17){clZero(lines[k].a[j]);lines[k].a[j].sign=sign;lines[k].a[j].limb[0]=word;}
        }
        if(any!=0u?odd==0u:exponent!=0u)return false;
        if(k>=17&&any==0u)return false; // constructor refuses planes through camera
    }
    return true;
}
uint clCharge(uint op){return 81920u+64u*(op==0u?177u:op==1u?301u:467u);}
void clEvaluate(CL_RECORD(recordWords),CL_REQUEST(requestWords),CL_RESULT(resultWords)){
    for(int k=0;k<99;k++)resultWords[k]=0u;
    for(int k=0;k<65;k++)resultWords[34+k]=requestWords[k];
    uint op=requestWords[58],charge=clCharge(op);resultWords[1]=charge;
    // Reject BEFORE preparation/arithmetic. No logical counter controls work.
    if(requestWords[57]!=1u||op>3u||requestWords[59]<charge||requestWords[64]!=0u)return;
    uint a=requestWords[60],b=requestWords[61],c=requestWords[62],d=requestWords[63];
    if(a>=17u||b>=17u||c>=17u||d>=17u||(op==0u&&(c!=0u||d!=0u))||(op==1u&&d!=0u))return;
    CLLine lines[17];if(!clPrepare(recordWords,requestWords,lines))return;
    bool ok=true;CLPoint p,q;for(int k=0;k<3;k++){clZero(p.a[k]);clZero(q.a[k]);}
    uint classification=clCross(lines[a],lines[b],a,b,p,ok),sign=0u;
    if(op==1u){if(classification==3u||classification==4u)return;sign=clIncidence(lines[c],p,ok);}
    if(op>=2u){uint second=clCross(lines[c],lines[d],c,d,q,ok);
        if(classification!=1u||second!=1u)return;sign=clCompare(p,q,op==2u?0:1,ok);}
    if(!ok)return;
    resultWords[0]=1u;resultWords[2]=classification;resultWords[3]=sign;
    for(int k=0;k<3;k++){
        resultWords[4+5*k]=p.a[k].sign;resultWords[19+5*k]=q.a[k].sign;
        for(int j=0;j<4;j++){resultWords[5+5*k+j]=p.a[k].limb[j];resultWords[20+5*k+j]=q.a[k].limb[j];}
    }
}
)CAMERALINE";
}
inline std::string languageSource(bool metal)
{
    // Import the accepted stateless split16 word functions unchanged. No cl*
    // operation calls a 35-limb EL/CC routine; the constructor is probe-only.
    auto source=cameraraycoefficients::languageSource(metal);
    source+=metal?"\n#define CL_RECORD(N) thread const uint (&N)[557]\n#define CL_REQUEST(N) thread const uint (&N)[65]\n#define CL_RESULT(N) thread uint (&N)[99]\n#define CL_LINES(N) thread CLLine (&N)[17]\n":
        "\n#define CL_RECORD(N) uint N[557]\n#define CL_REQUEST(N) uint N[65]\n#define CL_RESULT(N) inout uint N[99]\n#define CL_LINES(N) inout CLLine N[17]\n";
    return source+arithmeticSource();
}
inline std::string probeBody()
{
    return "uint a[65],r[557],b[99];for(int k=0;k<65;k++)a[k]=inputs[i*623u+uint(k)];"
        "uint mode=inputs[i*623u+65u];if(mode==0u){uint c[57];for(int k=0;k<57;k++)c[k]=a[k];ccEvaluate(c,r);}"
        "else{for(int k=0;k<557;k++)r[k]=mode==1u?inputs[i*623u+66u+uint(k)]:0u;}"
        "clEvaluate(r,a,b);for(int k=0;k<99;k++)outputs[i*99u+uint(k)]=b[k];";
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
        "\nkernel void cameraLinePredicateProbe(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],"
        "uint i [[thread_position_in_grid]]){"+probeBody()+"}\n";
}
} // namespace arbitgpu::cameraline
