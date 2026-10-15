#pragma once
#include "fixture_native_exact_line_predicates.h"
#include <array>
#include <cstdint>
#include <string>

namespace arbitgpu::cameraraycoefficients
{
// Pure mathematical construction over supplied captured binary32 bitwords.
// No uniform capture, native owner authentication, visibility or coverage grant.
// The receiver parent anchors an emitted grating-order hit and separate raster
// center attribution. It does not contain a native camera-center ray query.
// Backend camera-ray intervals/conventions remain UNBOUND. Projector-depth clip
// forms below do not prove any Euclidean query-distance interval. A radial bound
// H²(D·D)<=maximum²*n² is generally quadratic and needs a separate certificate.
// This exact model interprets uploaded values algebraically. It does not replay
// rounded normalization, intersection arithmetic or hardware raster edge rules.
constexpr std::uint32_t kVersion=1,kRequestWords=57,kLineWords=28,kLineCount=17;
constexpr std::uint32_t kResultWords=4+kRequestWords+kLineCount*kLineWords+2*10;
using Request=std::array<std::uint32_t,kRequestWords>;
using Result=std::array<std::uint32_t,kResultWords>;
static_assert(sizeof(Request)==228&&sizeof(Result)==2228,"coefficient wire layout");
// Request: version, precision domain 30 or 256, admitted visit limit, reserved=0;
// frame tags[4], pixel x/y integer <=2^24; camera xyz[3], quaternion xyzw[4],
// projection tangent/aspect/near/far[4], extent width/height[2]. Then two triangle
// records of 17 words: table/object/material/instance/triangle/vertexA/B/C tags[8]
// and xyz bitwords for A/B/C[9]. Tags are opaque lineage, never native proof.
// Captured tangent/aspect/extents/near must be positive, far>near. Pixel is inside
// captured extent. No exact-unit quaternion requirement. A singular camera ray
// map, collinear triangle or triangle plane through camera is Unsupported.
// Result: constructed=1 or Unsupported=0, full admitted visits, performed visits,
// precision; exact request copy; 17 forms; two scalar H values. Failed construction
// zeros all mathematical output, keeping admitted/performed counts only.
// Each form stores signed exponent, then three [sign,8 low uint32 limbs] values.
// The ray D here is width*height*R_q(filmRay), a common positive extent
// denominator scale. Film coordinates use pixel+(u,v), top-first y. Its
// normalization cancels from plane-hit positions, edge ratios and same-ray depth.
// Coefficients are integers times ONE common positive 2^exponent, ordered u,v,1
// over the nominated top-first pixel. Zero has sign=0, mantissa=0, exponent=0.
// Signs are 0/1/2 for zero/positive/negative. No float conversion of generated data.
// Forms 0..2 are world D_x/D_y/D_z, 3 is G·D; 4..9 and 10..15 are triangle
// AB/BC/CA edges, horizon n, PROJECTOR near/far clip; 16 is depth exchange pair.
// H scalar stores exponent/sign/8 limbs. Edges refer to exact ordered endpoint
// tags/positions in the copied triangle record, not an infinite-line-only seam.
// Closed edge membership is edge/n>=0, forward intersection H/n>0. Reversing
// winding flips edge, H and n together. Projector depth is H*(G·D)/n; clip signs
// must account for n. Pair is H0*n1-H1*n0; ordering must account for n0*n1.
// A zero pair form means exact equal plane-ray depths, with NO invented tie key.

constexpr int kMinimumExponent=-512,kMaximumExponent=512;
// Canonical individual dyadic mantissas and common-line aligned mantissas must
// fit the selected 30/256-bit domain. Inputs have <=24 significand bits, exponents
// -149..127. Every multiplication starts with bounded odd mantissas: product
// <=2P bits. Addition aligns only if each shifted operand fits P bits, then its
// intermediate sum has <=P+1 bits. Normalize before checking the P-bit result.
// Exponent products lie in [-1024,1024]; normalization adds at most 2P-1. All
// signed exponent intermediates fit [-1024,1535], then refuse outside [-512,512].
// Alignment distance <=1024 is checked BEFORE any shift; excessive precision
// refuses. This is an explicit bounded domain, not the full finite-f32 algebra.
// Generated line |coefficient|<2^256 gives cross <2^513, det3 <2^771, event
// comparison <2^1027, fitting the existing 35-limb core. Raw-f32 API is unchanged.
// For <=30-bit coefficients det2 can need 61 bits, hence TWO limbs. Arbitrary
// full 32-bit coefficients can need 65 bits, hence THREE, not two.
constexpr std::uint32_t kAddVisits=2*137+175+169;
constexpr std::uint32_t multiplyVisits(std::uint32_t precision)
{return exactline::multiplyVisits(precision==30?1:8,precision==30?1:8)+169;}
// Full constructor: <=223 products,167 sums;35 decoded constants/inputs,17 form
// conversions,9 explicit zero dyadics and2 scalar stores. Width scans=35+32, shift=67+35+35=137,
// normalization=35+32+32+35+35=169. All EL clears/capacity/carry scans remain 35.
// Product active width never makes those loops cheap. Failed work keeps the full
// admitted envelope; insufficient admission refuses BEFORE decode/construction.
constexpr std::uint32_t maximumVisits(std::uint32_t precision)
{return precision==30||precision==256?223*multiplyVisits(precision)+167*kAddVisits+35*204+9*35+17*442+20:0;}
static_assert(maximumVisits(30)==210740&&maximumVisits(256)==1208219,"construction visit envelopes");
// Prospective source-cost envelope, not measured GPU instructions or a lower bound.
// The223 multiply/167 add census includes83 subtract wrappers,39 dots,9 crosses
// and10 vector subtractions. Other calls:35 decode,425 normalize,385 shift/width,
// 17 form stores,2 scalar stores,5 negative,1 entry. Zero calls are bounded by
// 44 explicit plus390 possibly failed multiply/add calls =434. Their failure
// zeros replace larger success paths; charging both call sets is conservative.
// Total CC calls<=2220. Reserve128 per call for straight-line scalar/index/setup
// work, plus32768 for fixed driver checks/loops. Reserve64 per worst-case visit
// for loop arithmetic and amortized EL setup, separately from copies. The larger
// EL sum body is amortized over its full175 clear/arithmetic/scan visits.
// Copy counts include full GLSL value/inout copies and cover Metal too:
// ccZero82,width45,normalize83,decode85,shift118,add/subtract/multiply157,
// negative74,vecSubtract/cross453,dot305,scalarStore1160,formStore1235.
// The stores include557-word result-buffer copy-in/out. Additionally each
// normalization may assign36 words, and each zero-operand add may assign37.
// Fixed8192 includes wrapper614+entry614+initial clear557+failure clear554
// +echo57+direct dyadic assignments481+metadata/work8+index tables18=2903
// moves, with room for fixed scalar/return moves. Charge8 per moved word for
// move/loop/index costs, even if the compiler removes copies. This multiplier
// does not establish a minimum operation count or an actual resource receipt.
constexpr std::uint64_t maximumWordMoves(std::uint32_t precision)
{
    if(precision!=30&&precision!=256)return 8192;
    const std::uint64_t l=precision==30?1:8,b=223*l*l;
    // Full EL calls:223 multiply,167 sum,1244 zero,836 capacity,390 normalize,
    // 167 magnitude,2b accumulate,b product,8V work moves and70(b+167) word calls.
    // Zero1244 =44 explicit+425 normalize+385 shift+223 multiply+167 sum;
    // capacity836 =3*223+167. Failed CC branches use fewer EL calls than these.
    return 8192+434*82+385*45+425*83+35*85+385*118+473*157+5*74+
        10*453+39*305+9*453+2*1160+17*1235+425*36+167*37+
        223*155+167*152+1244*80+836*45+390*80+167*81+
        164*b+8*maximumVisits(precision)+8*b+6*(70*b+70*167);
}
constexpr std::uint64_t maximumScalarOperations(std::uint32_t precision)
{return 32768+(maximumVisits(precision)?128ull*2220:0)+64ull*maximumVisits(precision)+8*maximumWordMoves(precision);}
static_assert(maximumWordMoves(30)==2414604&&maximumWordMoves(256)==18711444,"constructor nested copy schedules");
static_assert(maximumScalarOperations(30)==33121120&&maximumScalarOperations(256)==227334496
    &&maximumScalarOperations(0)==98304,"complete attempted constructor source costs");
// Counter saturation is not physical cessation; retain the full failed-attempt
// cost. Invalid precision charges setup/clears too. Transfers, host/caller copies,
// compiled spills/pipelines and retained native owners need separate admission.
// This remains a two-triangle request, with no geometry-work conversion or full
// arrangement affordability grant. No logical scratch increase is needed.
// <=192 simultaneously live dyadics/value-parameter/EL temporaries, each148 bytes;
// include two request and three result copies and1024 scalar bytes. This is logical scratch,
// not measured native program/storage. Full arrangement admission remains OPEN.
constexpr std::uint64_t kLogicalScratchBytes=192ull*148+2*kRequestWords*4+3*kResultWords*4+1024;
static_assert(kLogicalScratchBytes==36580,"logical constructor scratch");

inline const char* constructionSource()
{
return R"CAMERA(
struct CCDyadic { ELInt m; int exponent; };
struct CCVec { CCDyadic a[3]; };
void ccZero(EL_REF(CCDyadic,r),EL_REF(ELWork,w)){elZero(r.m,w);r.exponent=0;}
int ccWidth(EL_VAL(ELInt,a),EL_REF(ELWork,w)){
    int high=-1;for(int k=0;k<35;k++){elStep(w);if(a.limb[k]!=0u)high=k;}
    if(high<0)return 0;int bits=0;
    for(int k=0;k<32;k++){elStep(w);if((a.limb[high]>>uint(k))!=0u)bits=k+1;}
    return high*32+bits;
}
void ccNormalize(EL_REF(CCDyadic,r),uint bitLimit,EL_REF(ELWork,w)){
    if(!w.ok)return;int low=-1,high=-1;
    for(int k=0;k<35;k++){elStep(w);if(r.m.limb[k]!=0u){if(low<0)low=k;high=k;}}
    if(low<0){r.m.sign=0u;r.exponent=0;return;}
    int trailing=32,leading=0;
    for(int k=0;k<32;k++){elStep(w);if(trailing==32&&((r.m.limb[low]>>uint(k))&1u)!=0u)trailing=k;}
    for(int k=0;k<32;k++){elStep(w);if((r.m.limb[high]>>uint(k))!=0u)leading=k+1;}
    int shift=low*32+trailing,width=high*32+leading-shift;
    int exponent=r.exponent+shift;
    if(width>int(bitLimit)||exponent< -512||exponent>512){w.ok=false;return;}
    ELInt reduced;elZero(reduced,w);reduced.sign=r.m.sign;
    int words=shift/32,bits=shift%32;
    for(int k=0;k<35;k++){elStep(w);int at=k+words;
        if(at<35){reduced.limb[k]=r.m.limb[at]>>uint(bits);
            if(bits!=0&&at+1<35)reduced.limb[k]|=r.m.limb[at+1]<<uint(32-bits);}}
    r.m=reduced;r.exponent=exponent;
}
void ccDecode(uint word,bool rawFloat,EL_REF(CCDyadic,r),uint bitLimit,EL_REF(ELWork,w)){
    ccZero(r,w);if(!w.ok)return;
    uint significand=word;
    if(rawFloat){uint e=(word>>23u)&255u;if(e==255u){w.ok=false;return;}
        significand=word&8388607u;if(e!=0u)significand|=8388608u;
        r.exponent=e==0u? -149:int(e)-150;r.m.sign=(word>>31u)==0u?1u:2u;}
    else r.m.sign=1u;
    r.m.limb[0]=significand;ccNormalize(r,bitLimit,w);
}
void ccShift(EL_VAL(ELInt,a),int shift,uint bitLimit,EL_REF(ELInt,r),EL_REF(ELWork,w)){
    int width=ccWidth(a,w);elZero(r,w);
    if(!w.ok)return;
    if(shift<0||shift>int(bitLimit)||width+shift>int(bitLimit)){w.ok=false;return;}
    r.sign=a.sign;int words=shift/32,bits=shift%32;
    for(int k=0;k<35;k++){elStep(w);int at=k-words;
        if(at>=0){r.limb[k]=a.limb[at]<<uint(bits);
            if(bits!=0&&at>0)r.limb[k]|=a.limb[at-1]>>uint(32-bits);}}
}
void ccAdd(EL_VAL(CCDyadic,a),EL_VAL(CCDyadic,b),EL_REF(CCDyadic,r),uint bitLimit,EL_REF(ELWork,w)){
    if(!w.ok){ccZero(r,w);return;}if(a.m.sign==0u){r=b;return;}if(b.m.sign==0u){r=a;return;}
    int exponent=min(a.exponent,b.exponent);ELInt x,y;
    ccShift(a.m,a.exponent-exponent,bitLimit,x,w);ccShift(b.m,b.exponent-exponent,bitLimit,y,w);
    elSum(x,y,r.m,w);r.exponent=exponent;ccNormalize(r,bitLimit,w);
}
void ccSubtract(EL_VAL(CCDyadic,a),EL_VAL(CCDyadic,b),EL_REF(CCDyadic,r),uint bitLimit,EL_REF(ELWork,w)){
    if(b.m.sign!=0u)b.m.sign=3u-b.m.sign;ccAdd(a,b,r,bitLimit,w);
}
void ccMultiply(EL_VAL(CCDyadic,a),EL_VAL(CCDyadic,b),EL_REF(CCDyadic,r),uint bitLimit,EL_REF(ELWork,w)){
    if(!w.ok){ccZero(r,w);return;}
    uint limbs=bitLimit==30u?1u:8u;
    elMultiply(a.m,b.m,limbs,limbs,2u*limbs,r.m,w);r.exponent=a.exponent+b.exponent;
    ccNormalize(r,bitLimit,w);
}
void ccNegative(EL_REF(CCDyadic,a)){if(a.m.sign!=0u)a.m.sign=3u-a.m.sign;}
void ccVecSubtract(EL_VAL(CCVec,a),EL_VAL(CCVec,b),EL_REF(CCVec,r),uint p,EL_REF(ELWork,w)){
    for(int k=0;k<3;k++)ccSubtract(a.a[k],b.a[k],r.a[k],p,w);
}
void ccDot(EL_VAL(CCVec,a),EL_VAL(CCVec,b),EL_REF(CCDyadic,r),uint p,EL_REF(ELWork,w)){
    CCDyadic x,y,z,s;ccMultiply(a.a[0],b.a[0],x,p,w);ccMultiply(a.a[1],b.a[1],y,p,w);
    ccAdd(x,y,s,p,w);ccMultiply(a.a[2],b.a[2],z,p,w);ccAdd(s,z,r,p,w);
}
void ccCross(EL_VAL(CCVec,a),EL_VAL(CCVec,b),EL_REF(CCVec,r),uint p,EL_REF(ELWork,w)){
    for(int k=0;k<3;k++){int i=(k+1)%3,j=(k+2)%3;CCDyadic x,y;
        ccMultiply(a.a[i],b.a[j],x,p,w);ccMultiply(a.a[j],b.a[i],y,p,w);ccSubtract(x,y,r.a[k],p,w);}
}
void ccStoreScalar(EL_VAL(CCDyadic,a),int base,CC_BUFFER(resultWords),EL_REF(ELWork,w)){
    elStep(w);resultWords[base]=uint(a.exponent);elStep(w);resultWords[base+1]=a.m.sign;
    for(int k=0;k<8;k++){elStep(w);resultWords[base+2+k]=a.m.limb[k];}
}
void ccStoreForm(EL_VAL(CCVec,a),int index,CC_BUFFER(resultWords),uint p,EL_REF(ELWork,w)){
    int exponent=0;bool found=false;
    for(int k=0;k<3;k++){elStep(w);if(a.a[k].m.sign!=0u){if(!found||a.a[k].exponent<exponent)exponent=a.a[k].exponent;found=true;}}
    int base=61+index*28;elStep(w);resultWords[base]=uint(exponent);
    for(int k=0;k<3;k++){ELInt aligned;int shift=a.a[k].m.sign==0u?0:a.a[k].exponent-exponent;
        ccShift(a.a[k].m,shift,p,aligned,w);elStep(w);resultWords[base+1+k*9]=aligned.sign;
        for(int limb=0;limb<8;limb++){elStep(w);resultWords[base+2+k*9+limb]=aligned.limb[limb];}}
}
void ccEvaluate(CC_REQUEST(requestWords),CC_RESULT(resultWords)){
    for(int k=0;k<557;k++)resultWords[k]=0u;
    uint p=requestWords[1],required=CC_BOUND(p);
    if(requestWords[0]!=1u||required==0u||requestWords[2]<required||requestWords[3]!=0u
        ||requestWords[8]>16777216u||requestWords[9]>16777216u)return;
    resultWords[1]=required;ELWork w;w.used=0u;w.limit=required;w.capacity=35u;w.ok=true;
    CCVec camera,vertex[6];CCDyadic q[4],projection[4],extent[2],pixel[2],one,two;
    for(int k=0;k<3;k++)ccDecode(requestWords[10+k],true,camera.a[k],p,w);
    for(int k=0;k<4;k++){ccDecode(requestWords[13+k],true,q[k],p,w);ccDecode(requestWords[17+k],true,projection[k],p,w);}
    for(int k=0;k<2;k++){ccDecode(requestWords[21+k],true,extent[k],p,w);ccDecode(requestWords[8+k],false,pixel[k],p,w);}
    for(int tri=0;tri<2;tri++)for(int v=0;v<3;v++)for(int axis=0;axis<3;axis++)
        ccDecode(requestWords[31+tri*17+v*3+axis],true,vertex[tri*3+v].a[axis],p,w);
    ccDecode(1u,false,one,p,w);ccDecode(2u,false,two,p,w);
    CCDyadic difference;
    if(w.ok){ccSubtract(projection[3],projection[2],difference,p,w);
        if(projection[0].m.sign!=1u||projection[1].m.sign!=1u||projection[2].m.sign!=1u
            ||difference.m.sign!=1u||extent[0].m.sign!=1u||extent[1].m.sign!=1u)w.ok=false;
        // Pixel-vs-extent comparison uses exact dyadics, separately charged below.
        ccSubtract(extent[0],pixel[0],difference,p,w);if(difference.m.sign!=1u)w.ok=false;
        ccSubtract(extent[1],pixel[1],difference,p,w);if(difference.m.sign!=1u)w.ok=false;}
    CCVec film[3],rotation[3],direction[3],g,gd;CCDyadic wh,ta,h2,w2,x,y;
    if(w.ok){
        for(int i=0;i<3;i++)for(int j=0;j<3;j++)ccZero(film[i].a[j],w);
        ccMultiply(extent[0],extent[1],wh,p,w);ccMultiply(projection[0],projection[1],ta,p,w);
        ccMultiply(extent[1],two,h2,p,w);ccMultiply(extent[0],two,w2,p,w);
        ccMultiply(h2,ta,film[0].a[0],p,w);ccMultiply(w2,projection[0],film[1].a[1],p,w);ccNegative(film[1].a[1]);
        ccMultiply(h2,pixel[0],x,p,w);ccSubtract(x,wh,y,p,w);ccMultiply(y,ta,film[2].a[0],p,w);
        ccMultiply(w2,pixel[1],x,p,w);ccSubtract(wh,x,y,p,w);ccMultiply(y,projection[0],film[2].a[1],p,w);
        film[2].a[2]=wh;ccNegative(film[2].a[2]);
        CCDyadic products[9];int left[9],right[9];
        left[0]=0;right[0]=0;left[1]=1;right[1]=1;left[2]=2;right[2]=2;
        left[3]=0;right[3]=1;left[4]=0;right[4]=2;left[5]=1;right[5]=2;
        left[6]=3;right[6]=0;left[7]=3;right[7]=1;left[8]=3;right[8]=2;
        for(int k=0;k<9;k++)ccMultiply(q[left[k]],q[right[k]],products[k],p,w);
        for(int k=0;k<3;k++){
            int a=(k+1)%3,b=(k+2)%3;ccAdd(products[a],products[b],x,p,w);
            ccMultiply(x,two,y,p,w);ccSubtract(one,y,rotation[k].a[k],p,w);}
        ccSubtract(products[3],products[8],x,p,w);ccMultiply(x,two,rotation[0].a[1],p,w);
        ccAdd(products[3],products[8],x,p,w);ccMultiply(x,two,rotation[1].a[0],p,w);
        ccAdd(products[4],products[7],x,p,w);ccMultiply(x,two,rotation[0].a[2],p,w);
        ccSubtract(products[4],products[7],x,p,w);ccMultiply(x,two,rotation[2].a[0],p,w);
        ccSubtract(products[5],products[6],x,p,w);ccMultiply(x,two,rotation[1].a[2],p,w);
        ccAdd(products[5],products[6],x,p,w);ccMultiply(x,two,rotation[2].a[1],p,w);
        for(int column=0;column<3;column++)for(int row=0;row<3;row++)ccDot(rotation[row],film[column],direction[column].a[row],p,w);
        // R_conj(q)=transpose(R_q) algebraically, even for non-unit q. It is
        // NOT assumed to be its inverse. G is minus the authored conjugate row z.
        for(int k=0;k<3;k++){g.a[k]=rotation[k].a[2];ccNegative(g.a[k]);}
        for(int k=0;k<3;k++)ccDot(g,direction[k],gd.a[k],p,w);
        CCVec crossed;ccCross(direction[0],direction[1],crossed,p,w);ccDot(crossed,direction[2],difference,p,w);
        if(difference.m.sign==0u)w.ok=false;
    }
    CCDyadic heights[2];CCVec horizons[2],edges[6],nearClip[2],farClip[2],exchange;
    if(w.ok)for(int tri=0;tri<2;tri++){
        CCVec u,v,t,e1,e2,normal,edgeNormal;
        ccVecSubtract(vertex[tri*3],camera,u,p,w);ccVecSubtract(vertex[tri*3+1],camera,v,p,w);ccVecSubtract(vertex[tri*3+2],camera,t,p,w);
        ccVecSubtract(vertex[tri*3+1],vertex[tri*3],e1,p,w);ccVecSubtract(vertex[tri*3+2],vertex[tri*3],e2,p,w);
        ccCross(e1,e2,normal,p,w);ccDot(normal,u,heights[tri],p,w);
        if(heights[tri].m.sign==0u)w.ok=false;
        for(int k=0;k<3;k++)ccDot(normal,direction[k],horizons[tri].a[k],p,w);
        for(int edge=0;edge<3;edge++){
            if(edge==0)ccCross(u,v,edgeNormal,p,w);else if(edge==1)ccCross(v,t,edgeNormal,p,w);else ccCross(t,u,edgeNormal,p,w);
            for(int k=0;k<3;k++)ccDot(edgeNormal,direction[k],edges[tri*3+edge].a[k],p,w);}
        for(int k=0;k<3;k++){
            ccMultiply(heights[tri],gd.a[k],x,p,w);ccMultiply(projection[2],horizons[tri].a[k],y,p,w);ccSubtract(x,y,nearClip[tri].a[k],p,w);
            ccMultiply(projection[3],horizons[tri].a[k],y,p,w);ccSubtract(x,y,farClip[tri].a[k],p,w);}
    }
    if(w.ok){
        for(int k=0;k<3;k++){ccMultiply(heights[0],horizons[1].a[k],x,p,w);ccMultiply(heights[1],horizons[0].a[k],y,p,w);ccSubtract(x,y,exchange.a[k],p,w);}
        for(int row=0;row<3;row++){CCVec component;for(int k=0;k<3;k++)component.a[k]=direction[k].a[row];ccStoreForm(component,row,resultWords,p,w);}
        ccStoreForm(gd,3,resultWords,p,w);
        for(int tri=0;tri<2;tri++){
            for(int edge=0;edge<3;edge++)ccStoreForm(edges[tri*3+edge],4+tri*6+edge,resultWords,p,w);
            ccStoreForm(horizons[tri],7+tri*6,resultWords,p,w);ccStoreForm(nearClip[tri],8+tri*6,resultWords,p,w);ccStoreForm(farClip[tri],9+tri*6,resultWords,p,w);
            ccStoreScalar(heights[tri],537+tri*10,resultWords,w);}
        ccStoreForm(exchange,16,resultWords,p,w);
    }
    resultWords[2]=w.used;
    if(!w.ok){for(int k=3;k<557;k++)resultWords[k]=0u;return;}
    for(int k=0;k<57;k++)resultWords[4+k]=requestWords[k];resultWords[3]=p;resultWords[0]=1u;
}
)CAMERA";
}
inline std::string languageSource(bool metal)
{
    auto source=exactline::languageSource(metal);
    source+=metal?"\n#define CC_REQUEST(N) thread const uint* N\n#define CC_RESULT(N) thread uint* N\n#define CC_BUFFER(N) thread uint* N\n":
        "\n#define CC_REQUEST(N) uint N[57]\n#define CC_RESULT(N) out uint N[557]\n#define CC_BUFFER(N) inout uint N[557]\n";
    source+="uint ccBound(uint p){return p==30u?"+std::to_string(maximumVisits(30))+"u:p==256u?"
        +std::to_string(maximumVisits(256))+"u:0u;}\n#define CC_BOUND(P) ccBound(P)\n";
    source+=constructionSource();return source;
}
inline std::string glslProbeSource()
{
    return "#version 430 core\nlayout(local_size_x=1) in;\nlayout(std430,binding=0) readonly buffer Inputs {uint inputs[];};\n"
        "layout(std430,binding=1) writeonly buffer Outputs {uint outputs[];};\n"+languageSource(false)+
        "\nvoid main(){uint i=gl_GlobalInvocationID.x;uint a[57],b[557];for(int k=0;k<57;k++)a[k]=inputs[i*57u+uint(k)];"
        "ccEvaluate(a,b);for(int k=0;k<557;k++)outputs[i*557u+uint(k)]=b[k];}\n";
}
inline std::string metalProbeSource()
{
    return "#include <metal_stdlib>\nusing namespace metal;\n"+languageSource(true)+
        "\nkernel void cameraRayCoefficientProbe(device const uint* inputs [[buffer(0)]],device uint* outputs [[buffer(1)]],"
        "uint i [[thread_position_in_grid]]){uint a[57],b[557];for(int k=0;k<57;k++)a[k]=inputs[i*57u+uint(k)];"
        "ccEvaluate(a,b);for(int k=0;k<557;k++)outputs[i*557u+uint(k)]=b[k];}\n";
}
}
