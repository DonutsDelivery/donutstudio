#pragma once

#include "particle_rotating_exact_scalar.h"
#include <cstddef>

namespace videorender::rotatingplane
{
using rotatingexact::Integer;
// The remaining response phases borrow this same storage. No register table,
// product buffer, staging array or wide value is copied by the geometric leaf.
struct Workspace {
    std::array<Integer,76> integer;
    rotatingexact::Workspace scalar;
    std::array<float,32> staging;
    std::array<std::int32_t,32> control;
    std::array<double,8> receipt;
};
static_assert(sizeof(Workspace)==27768 && alignof(Workspace)==alignof(double));
static_assert(offsetof(Workspace,scalar)==26752 && offsetof(Workspace,staging)==27448
    && offsetof(Workspace,control)==27576 && offsetof(Workspace,receipt)==27704);
static_assert(std::is_trivial_v<Workspace> && std::is_standard_layout_v<Workspace>);

// Value coefficients only. A successful certificate issues no source, capture,
// body, history or replay capability. The caller must establish those separately.
struct Input {
    std::array<float,3> previousCenter,scheduledCenter,half;
    float reach,dt,plane;
    float previousZ,previousW,rawZ,rawW;
    float previousA,previousB,scheduledA,scheduledB;
    unsigned normalAxis;
};
struct Mesh {
    const float* positions;
    const std::uint32_t* indices;
    unsigned vertices,triangles;
};
enum class Kind : std::uint32_t { separated,impact };
struct Result {
    Kind kind;
    unsigned limbs,grid,normalAxis;
    int side;
    unsigned triangle,point,reserved;
};
static_assert(sizeof(Result)==32 && std::is_trivial_v<Result>);

// Literal first-three-phase ceilings from the accepted contract. The two
// response phases and replay/caller payers remain outside this prerequisite.
struct Cost {
    static constexpr bool supported(unsigned n) { return n>=22 && n<=87; }
    static constexpr unsigned grid(unsigned n)
    { return supported(n) ? ((32*n-684)/14<149 ? (32*n-684)/14:149):0; }
    static constexpr std::uint64_t first(unsigned n,unsigned vertices,unsigned triangles)
    {
        using C=rotatingexact::Cost;
        return supported(n) && vertices<=256 && triangles<=128
            ? 20*C::decode(n)+15*C::multiply(n)+24*C::add(n)+16*C::compare(n)
                +13*C::shift(n)+84+3*vertices+3*triangles:C::unavailable;
    }
    static constexpr std::uint64_t face(unsigned n)
    {
        using C=rotatingexact::Cost;
        return supported(n) ? 28*C::multiply(n)+36*C::add(n)+8*C::compare(n)
            +8*C::shift(n)+18:C::unavailable;
    }
    static constexpr std::uint64_t triangle(unsigned n)
    {
        using C=rotatingexact::Cost;
        return supported(n) ? 6*C::decode(n)+39*C::multiply(n)+54*C::add(n)
            +31*C::compare(n):C::unavailable;
    }
};
inline bool supportedFloat(float value)
{
    const auto bits=rotatingexact::floatBits(value)&0x7fffffffU;
    return !bits || (bits>=0x00800000U && bits<0x7f800000U);
}
inline bool bounded(float value,std::uint32_t absoluteMaximum)
{ return supportedFloat(value) && (rotatingexact::floatBits(value)&0x7fffffffU)<=absoluteMaximum; }
inline bool gridFits(float value,unsigned grid)
{
    const auto bits=rotatingexact::floatBits(value)&0x7fffffffU;
    if (!bits) return true;
    if (!supportedFloat(value)) return false;
    const auto mantissa=(bits&0x7fffffU)|0x800000U;
    return int(bits>>23)-150+int(rotatingexact::trailingZeros(mantissa))+int(grid)>=0;
}
inline bool sameFloat(float a,float b)
{
    const auto x=rotatingexact::floatBits(a),y=rotatingexact::floatBits(b);
    return x==y || (!(x&0x7fffffffU) && !(y&0x7fffffffU));
}
// One n-word sign/zero reduction is a C visit. It also rejects malformed signs.
inline bool sign(const Integer& value,unsigned n,int& result)
{
    if (!rotatingexact::widthSupported(n) || value.negative>1) return false;
    std::uint32_t any=0;
    for (unsigned i=0;i<n;++i) any|=value.word[i];
    result=any ? (value.negative ? -1:1):0;
    return true;
}
inline void negate(Integer& value) { value.negative^=1; }

// Register ownership at entry: 0..19 inputs, 20..21 root, 22..23
// endpoint norms, 24..30 endpoint factors, 31..34 root quaternion,
// 35..36 rigid pair, 37..39 impact axes, 40..42 center, 43..46 local
// products, 47..56 candidates, 57..62 triangle, 63..67 selected factors,
// 68..75 operations. Triangle streaming reuses freed local products 43..44
// for A's coordinates. The 76-register and 27768-byte peaks never increase.
// Exact output remains in the borrowed workspace. Result publishes only after
// all source triangles have a witness or a strict edge-separation proof.
template<class Budget>
inline bool certificate(const Input& input,const Mesh& mesh,unsigned n,Workspace& workspace,
                        Budget& budget,Result& output)
{
    using namespace rotatingexact;
    if (!rotatingplane::Cost::supported(n) || !mesh.positions || !mesh.indices || !mesh.vertices
        || !mesh.triangles || mesh.vertices>256 || mesh.triangles>128 || input.normalAxis>2)
        return false;
    const unsigned g=rotatingplane::Cost::grid(n),k=input.normalAxis,i=(k+1)%3,j=(k+2)%3;
    const auto overlaps=[&workspace](const void* data,std::size_t bytes) {
        const auto a=reinterpret_cast<std::uintptr_t>(data),b=reinterpret_cast<std::uintptr_t>(&workspace);
        return a>=b ? a-b<sizeof(Workspace):b-a<bytes;
    };
    if (!budget.charge(rotatingplane::Cost::first(n,mesh.vertices,mesh.triangles))) return false;
    if (overlaps(mesh.positions,3*mesh.vertices*sizeof(float))
        || overlaps(mesh.indices,3*mesh.triangles*sizeof(std::uint32_t))) return false;
    // Float-bit gates have bounded scalar visits. Decode performs each actual
    // fixed-grid inspection before writes. No float arithmetic defines a proof.
    if (!bounded(input.dt,0x3c088889U) || (floatBits(input.dt)>>31)
        || !(floatBits(input.dt)&0x7fffffffU) || !bounded(input.plane,0x42000000U)
        || !bounded(input.previousZ,0x3f800000U) || !bounded(input.previousW,0x3f800000U)
        || !bounded(input.rawZ,0x40000000U) || !bounded(input.rawW,0x40000000U)
        || !bounded(input.previousA,0x40a00000U) || !bounded(input.previousB,0x40a00000U)
        || !bounded(input.scheduledA,0x40a00000U) || !bounded(input.scheduledB,0x40a00000U)
        || !supportedFloat(input.reach)
        || ((floatBits(input.reach)&0x7fffffffU) && (floatBits(input.reach)>>31))) return false;
    const auto commonSign=[](float a,float b) {
        const auto x=floatBits(a),y=floatBits(b);
        return (x&0x7fffffffU) && (y&0x7fffffffU) && (x>>31)==(y>>31);
    };
    if (!commonSign(input.previousZ,input.rawZ) && !commonSign(input.previousW,input.rawW)) return false;
    for (unsigned axis=0;axis<3;++axis)
        if (!supportedFloat(input.previousCenter[axis]) || !supportedFloat(input.scheduledCenter[axis])
            || !supportedFloat(input.half[axis]) || (floatBits(input.half[axis])>>31)
            || !(floatBits(input.half[axis])&0x7fffffffU)) return false;
    for (unsigned vertex=0;vertex<mesh.vertices;++vertex) {
        const auto* position=mesh.positions+3*vertex;
        if (!bounded(position[i],0x42000000U) || !bounded(position[j],0x42000000U)
            || !gridFits(position[i],g) || !gridFits(position[j],g)
            || !supportedFloat(position[k]) || !sameFloat(position[k],input.plane)) return false;
    }
    for (unsigned index=0;index<3*mesh.triangles;++index)
        if (mesh.indices[index]>=mesh.vertices) return false;
    auto& r=workspace.integer;
    auto& product=workspace.scalar;
    const auto mul=[&](unsigned out,unsigned a,unsigned b) { return multiply(r[out],r[a],r[b],n,product); };
    const auto addAt=[&](unsigned out,unsigned a,unsigned b) { return add(r[out],r[a],r[b],n); };
    const auto sub=[&](unsigned out,unsigned a,unsigned b) { return subtract(r[out],r[a],r[b],n); };
    const auto shift=[&](unsigned out,unsigned a,unsigned bits) { return shiftLeft(r[out],r[a],n,bits); };
    // Scalar array only, no wide input copy or unpriced register initialization.
    workspace.staging[0]=input.previousCenter[i];
    workspace.staging[1]=input.previousCenter[j];
    workspace.staging[2]=input.previousCenter[k];
    workspace.staging[3]=input.scheduledCenter[i];
    workspace.staging[4]=input.scheduledCenter[j];
    workspace.staging[5]=input.scheduledCenter[k];
    workspace.staging[6]=input.half[i]; workspace.staging[7]=input.half[j]; workspace.staging[8]=input.half[k];
    workspace.staging[9]=input.reach; workspace.staging[10]=input.dt; workspace.staging[11]=input.plane;
    workspace.staging[12]=input.previousZ; workspace.staging[13]=input.previousW;
    workspace.staging[14]=input.rawZ; workspace.staging[15]=input.rawW;
    workspace.staging[16]=input.previousA; workspace.staging[17]=input.previousB;
    workspace.staging[18]=input.scheduledA; workspace.staging[19]=input.scheduledB;
    for (unsigned a=0;a<20;++a) if (!decodeFloat(r[a],workspace.staging[a],n,g)) return false;
    // Each endpoint: 5 M, 4 A, 6 Hs including its two residual bounds.
    for (unsigned endpoint=0;endpoint<2;++endpoint) {
        const unsigned z=12+2*endpoint,w=z+1,norm=22+endpoint,residual=24+2*endpoint;
        if (!mul(68,z,z) || !mul(69,w,w) || !addAt(norm,68,69)
            || !sub(70,69,68) || !mul(71,z,w) || !shift(71,71,1)
            || !mul(residual,16+2*endpoint,norm) || !shift(70,70,g) || !sub(residual,residual,70)
            || !mul(residual+1,17+2*endpoint,norm) || !shift(71,71,g)
            || !sub(residual+1,residual+1,71) || !shift(28+endpoint,norm,g)) return false;
        for (unsigned component=0;component<2;++component) {
            int comparison;
            if (!shift(72,residual+component,18)) return false;
            r[72].negative=0;
            if (!compare(r[72],r[28+endpoint],n,comparison) || comparison>0) return false;
        }
    }
    // Factored common endpoint residuals and H=S*N0*N1. 5 M and 1 Hs.
    if (!mul(24,24,23) || !mul(25,25,23) || !mul(26,26,22) || !mul(27,27,22)
        || !mul(30,22,23) || !shift(30,30,g)) return false;
    int sideComparison;
    if (!compare(r[2],r[11],n,sideComparison)) return false;
    const int sigma=sideComparison<0 ? -1:1;
    if (!sub(20,2,11) || !sub(68,5,11)) return false;
    if (sigma<0) { negate(r[20]); negate(r[68]); }
    if (!sub(20,20,8) || !sub(20,20,9) || !sub(68,68,8) || !sub(68,68,9)) return false;
    int initial,terminal;
    if (!sign(r[20],n,initial) || !sign(r[68],n,terminal) || initial<0) return false;
    if (terminal>0) {
        // Whole-plane distance proves separation, but still validate each
        // finite source triangle. This branch never pays the impact-face phase.
        for (unsigned triangle=0;triangle<mesh.triangles;++triangle) {
            if (!budget.charge(rotatingplane::Cost::triangle(n))) return false;
            for (unsigned vertex=0;vertex<3;++vertex) {
                const auto* position=mesh.positions+3*mesh.indices[3*triangle+vertex];
                if (!decodeFloat(r[57+2*vertex],position[i],n,g)
                    || !decodeFloat(r[58+2*vertex],position[j],n,g)) return false;
            }
            int areaSign;
            if (!sub(68,59,57) || !sub(69,60,58) || !sub(70,61,57) || !sub(71,62,58)
                || !mul(72,68,71) || !mul(73,69,70) || !sub(74,72,73)
                || !sign(r[74],n,areaSign) || !areaSign) return false;
        }
        output={Kind::separated,n,g,k,sigma,0,0,0}; return true;
    }
    if (!initial && !terminal) return false;
    if (!sub(21,20,68)) return false;

    if (!budget.charge(rotatingplane::Cost::face(n))) return false;
    // X/W at n/d: 4 M, 4 A. N and rigid pair: 3 M, 2 A, 1 Hs.
    if (!sub(68,14,12) || !mul(69,68,20) || !mul(31,12,21) || !addAt(31,31,69)
        || !sub(68,15,13) || !mul(69,68,20) || !mul(32,13,21) || !addAt(32,32,69)
        || !mul(68,31,31) || !mul(69,32,32) || !addAt(33,68,69)
        || !sub(35,69,68) || !mul(36,31,32) || !shift(36,36,1)) return false;
    // Shared Hd and N*H avoid duplicating denominator products: 11 M, 6 A.
    if (!mul(68,30,21) || !mul(39,33,68)) return false;
    for (unsigned axis=0;axis<2;++axis) {
        if (!mul(37+axis,35+axis,68) || !sub(69,26+axis,24+axis)
            || !mul(69,69,20) || !mul(70,24+axis,21) || !addAt(69,69,70)
            || !mul(69,69,33) || !addAt(37+axis,37+axis,69)) return false;
    }
    if (!mul(34,33,30)) return false;
    // In-plane C(t) uses denominator S*d; after multiplying N*H it uses
    // L=S*D, as do every actual face candidate and streamed triangle vertex.
    for (unsigned axis=0;axis<2;++axis)
        if (!sub(68,3+axis,axis) || !mul(69,68,20) || !mul(40+axis,axis,21)
            || !addAt(40+axis,40+axis,69) || !mul(40+axis,40+axis,34)) return false;
    if (!addAt(42,8,9)) return false;
    if (sigma<0) negate(r[42]);
    if (!addAt(42,42,11) || !mul(43,6,37) || !mul(44,6,38)
        || !mul(45,7,37) || !mul(46,7,38) || !shift(47,40,0) || !shift(48,41,0)
        || !shift(67,39,g)) return false;
    unsigned candidate=1;
    // Enumerate actual vertex bits so the y-plane basis z,x keeps source order.
    for (unsigned vertex=0;vertex<8;++vertex) {
        if (bool(vertex&(1U<<k))!=(sigma<0)) continue;
        const bool plusI=vertex&(1U<<i),plusJ=vertex&(1U<<j);
        const unsigned x=47+2*candidate,y=x+1;
        if (!(plusI ? addAt(x,40,43):sub(x,40,43))
            || !(plusJ ? sub(x,x,46):addAt(x,x,46))
            || !(plusI ? addAt(y,41,44):sub(y,41,44))
            || !(plusJ ? addAt(y,y,45):sub(y,y,45))) return false;
        ++candidate;
    }
    // Every actual contacting point has normal coordinate p+sigma*reach.
    // Its projected triangle point has normal coordinate p in input slot11.
    if (!(sigma>0 ? addAt(45,11,9):sub(45,11,9))) return false;
    // Above: exactly 28 M, 35 A, 4 Hs, no C or wide copies.
    bool selected=false;
    unsigned selectedTriangle=0,selectedPoint=0;
    for (unsigned triangle=0;triangle<mesh.triangles;++triangle) {
        if (!budget.charge(rotatingplane::Cost::triangle(n))) return false;
        // Decode six coordinates and form unscaled AB, BC, CA with six A
        // calls. Only A is scaled by D. Keeping edges unscaled preserves
        // the contract's 12g+557 determinant bound, rather than squaring D.
        const unsigned slots[6]={43,44,57,58,59,60};
        for (unsigned vertex=0;vertex<3;++vertex) {
            const auto* position=mesh.positions+3*mesh.indices[3*triangle+vertex];
            if (!decodeFloat(r[slots[2*vertex]],position[i],n,g)
                || !decodeFloat(r[slots[2*vertex+1]],position[j],n,g)) return false;
        }
        if (!sub(61,59,57) || !sub(62,60,58) || !sub(57,57,43) || !sub(58,58,44)
            || !sub(59,43,59) || !sub(60,44,60)) return false;
        const unsigned area=selected ? 75:66;
        if (!mul(70,58,59) || !mul(71,57,60) || !sub(area,70,71)
            || !mul(area,area,39) || !mul(43,43,39) || !mul(44,44,39)) return false;
        int winding;
        if (!sign(r[area],n,winding) || !winding) return false;
        bool contained=false;
        for (unsigned point=0;point<5;++point) {
            if (!sub(68,47+2*point,43) || !sub(69,48+2*point,44)) return false;
            bool inside=true;
            for (unsigned edge=0;edge<3;++edge) {
                const unsigned ex=edge==0 ? 57:(edge==1 ? 61:59),ey=ex+1;
                const unsigned determinant=selected ? 72:63+edge;
                if (!mul(70,ex,69) || !mul(71,ey,68) || !sub(determinant,70,71)
                    || (edge==1 && !addAt(determinant,determinant,area))) return false;
                int oriented;
                if (!sign(r[determinant],n,oriented)) return false;
                oriented*=winding;
                workspace.control[3*point+edge]=oriented;
                inside=inside && oriented>=0;
                if (winding<0) negate(r[determinant]);
            }
            if (inside) {
                contained=true;
                if (!selected) { selected=true; selectedTriangle=triangle; selectedPoint=point; }
            }
        }
        if (!contained) {
            bool separated=false;
            for (unsigned edge=0;edge<3;++edge) {
                bool outside=true;
                for (unsigned point=1;point<5;++point)
                    outside=outside && workspace.control[3*point+edge]<0;
                separated=separated || outside;
            }
            if (!separated) return false;
        }
        // Until this triangle ends its signed area supplies the BC offset.
        // The selected area is then positive and remains untouched thereafter.
        if (selected && selectedTriangle==triangle) r[66].negative=0;
        // 35 M, 37 A, 16 C and at most 15 scalar sign tests for separation
        // are below the literal 39 M,54 A,31 C ceiling. No Hs/copy/reset.
    }
    if (!selected) return false;
    output={Kind::impact,n,g,k,sigma,selectedTriangle,selectedPoint,0};
    return true;
}
// For impact results: root=integer[20]/integer[21], axes=37,38 over39,
// C_i/C_j and candidates=40,41 and47..56 over67. C_k=42/2^grid.
// Actual face-point normal=45/2^grid; projected normal=input11/2^grid.
// Selected oriented AB/BC/CA determinants=63..65, positive area=66.
// Their common geometric denominator is 2^(2*grid)*integer[39].
// Barycentrics A/B/C use BC/CA/AB respectively over66. All are exact.
}
