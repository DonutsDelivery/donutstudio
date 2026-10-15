#pragma once

#include "particle_rotating_plane_certificate.h"
#include <algorithm>
#include <cmath>

namespace videorender::rotatingplane
{
struct ResponseInput {
    std::array<float,3> velocity,angular,scheduledAngular;
    float mass,restitution,friction;
};
struct Response {
    Kind kind;
    unsigned triangle,point;
    std::array<float,3> center,velocity,angular;
    double time;
    std::array<double,3> weights,pointOnTriangle,impulse;
};
static_assert(sizeof(ResponseInput)==48 && sizeof(Response)==128
    && offsetof(Response,time)==48 && offsetof(Response,impulse)==104);
static_assert(std::is_trivial_v<Response> && std::is_standard_layout_v<Response>);
struct ResponseCost {
    static constexpr std::uint64_t rate(unsigned n) {
        using C=rotatingexact::Cost;
        return Cost::supported(n) ? 22*C::multiply(n)+20*C::add(n)+10*C::compare(n)
            +8*C::shift(n)+15*C::convert(n)+36:C::unavailable;
    }
    static constexpr std::uint64_t guard(unsigned n) {
        using C=rotatingexact::Cost;
        return Cost::supported(n) ? 2*C::decode(n)+6*C::add(n)+2*C::compare(n)+8:C::unavailable;
    }
    static constexpr std::uint64_t remainder(unsigned n) {
        using C=rotatingexact::Cost;
        return Cost::supported(n) ? 6*C::decode(n)+9*C::multiply(n)+24*C::add(n)
            +12*C::compare(n)+12*C::shift(n)+4*C::convert(n)+28+2*guard(n):C::unavailable;
    }
    // Contract W(n,V,T,P) for one body interval: every exact phase once, each
    // triangle once, and 4P+14 plane guards including remainder's two.
    static constexpr std::uint64_t interval(unsigned n,std::uint64_t vertices,std::uint64_t triangles,
                                            std::uint64_t points)
    {
        using C=rotatingexact::Cost;
        if (!Cost::supported(n) || vertices>256 || triangles>128 || points>64) return C::unavailable;
        return Cost::first(n,unsigned(vertices),unsigned(triangles))+Cost::face(n)+triangles*Cost::triangle(n)
            +rate(n)+remainder(n)+(4*points+12)*guard(n);
    }
};

// Select a supported successor using only the exact conversion's direction.
// The caller still decodes and compares the selected value on its fixed grid.
inline bool supportedUpper(float rounded,bool roundedAbove,float& output)
{
    using namespace rotatingexact;
    auto bits=floatBits(rounded);
    const auto absolute=bits&0x7fffffffU;
    if (!absolute) bits=0;
    if (absolute>=0x7f800000U) return false;
    if (absolute && absolute<0x00800000U) bits=(bits>>31) ? 0U:0x00800000U;
    else if (!roundedAbove) {
        if (bits>>31) bits=absolute==0x00800000U ? 0U:bits-1;
        else bits=absolute ? bits+1:0x00800000U;
    }
    if ((bits&0x7fffffffU)>=0x7f800000U) return false;
    output=floatFromBits(bits); return true;
}

// Pure coefficient response only. No capture is issued or consumed here.
// All floats and seven converted receipt scalars stay in borrowed staging until
// the exact remainder and its rounded endpoint have both passed their guards.
template<class Budget>
inline bool response(const Input& input,const Mesh& mesh,const ResponseInput& state,unsigned n,
                     Workspace& workspace,Budget& budget,Response& output)
{
    using namespace rotatingexact;
    const auto overlaps=[&workspace](const void* data,std::size_t bytes) {
        const auto a=reinterpret_cast<std::uintptr_t>(data),b=reinterpret_cast<std::uintptr_t>(&workspace);
        return a>=b ? a-b<sizeof(Workspace):b-a<bytes;
    };
    if (overlaps(&input,sizeof(input)) || overlaps(&state,sizeof(state))
        || overlaps(&mesh,sizeof(mesh)) || overlaps(&budget,sizeof(Budget))
        || overlaps(&output,sizeof(output))) return false;
    const auto outputOverlaps=[&output](const void* data,std::size_t bytes) {
        const auto a=reinterpret_cast<std::uintptr_t>(data),b=reinterpret_cast<std::uintptr_t>(&output);
        return a>=b ? a-b<sizeof(Response):b-a<bytes;
    };
    if (outputOverlaps(&input,sizeof(input)) || outputOverlaps(&state,sizeof(state))
        || outputOverlaps(&mesh,sizeof(mesh))
        || (mesh.vertices<=256 && outputOverlaps(mesh.positions,3*mesh.vertices*sizeof(float)))
        || (mesh.triangles<=128 && outputOverlaps(mesh.indices,3*mesh.triangles*sizeof(std::uint32_t)))) return false;
    Result geometry;
    if (!certificate(input,mesh,n,workspace,budget,geometry)) return false;
    if (geometry.kind!=Kind::impact) {
        output={Kind::separated,0,0,input.scheduledCenter,state.velocity,state.angular,0,{},{},{}};
        return true;
    }
    if (!budget.charge(ResponseCost::rate(n))) return false;
    if (!std::isfinite(state.mass) || state.mass<=0 || !std::isfinite(state.restitution)
        || state.restitution<0 || state.restitution>1 || !std::isfinite(state.friction)
        || state.friction<0 || state.friction>1) return false;
    for (unsigned a=0;a<3;++a)
        if (!std::isfinite(state.velocity[a]) || std::abs(state.velocity[a])>32
            || !std::isfinite(state.angular[a]) || std::abs(state.angular[a])>32
            || !std::isfinite(state.scheduledAngular[a]) || std::abs(state.scheduledAngular[a])>32)
            return false;
    const unsigned g=geometry.grid,k=geometry.normalAxis,i=(k+1)%3,j=(k+2)%3;
    const int sigma=geometry.side;
    for (unsigned a=0;a<3;++a) if (a!=k && state.scheduledAngular[a]!=0) return false;
    auto& r=workspace.integer;
    auto& s=workspace.staging;
    const auto mul=[&](unsigned out,unsigned a,unsigned b) { return multiply(r[out],r[a],r[b],n,workspace.scalar); };
    const auto addAt=[&](unsigned out,unsigned a,unsigned b) { return add(r[out],r[a],r[b],n); };
    const auto sub=[&](unsigned out,unsigned a,unsigned b) { return subtract(r[out],r[a],r[b],n); };
    const auto shift=[&](unsigned out,unsigned a,unsigned bits) { return shiftLeft(r[out],r[a],n,bits); };
    const auto narrow=[&](unsigned numerator,unsigned denominator,unsigned slot) {
        bool above; return convertFloat(r[numerator],r[denominator],n,r[72],s[slot],above);
    };
    const auto receipt=[&](unsigned numerator,unsigned denominator,unsigned slot) {
        bool above; return convertDouble(r[numerator],r[denominator],n,r[72],workspace.receipt[slot],above);
    };
    if (!receipt(64,66,1) || !receipt(65,66,2) || !receipt(63,66,3)) return false;
    // J=deltaZ*W-deltaW*X; the derivative denominator is N*N*H.
    if (!sub(57,14,12) || !sub(58,15,13) || !mul(68,57,32) || !mul(69,58,31)
        || !sub(59,68,69) || !mul(60,33,33) || !mul(68,31,32)
        || !mul(61,68,59) || !mul(61,61,21) || !shift(61,61,2)) return false;
    negate(r[61]);
    if (!mul(62,35,59) || !mul(62,62,21) || !shift(62,62,1)) return false;
    for (unsigned component=0;component<2;++component)
        if (!sub(68,26+component,24+component) || !mul(68,68,60)
            || !mul(61+component,61+component,30) || !addAt(61+component,61+component,68)) return false;
    if (!mul(70,60,30) || !mul(65,70,10)) return false;
    // The selected local signs follow the same world vertex-bit ordering.
    const unsigned corner=geometry.point ? geometry.point-1:0;
    const bool plusI=corner&(k==1 ? 2U:1U),plusJ=corner&(k==1 ? 1U:2U);
    if (!sub(75,6,6)) return false;
    const unsigned u=geometry.point ? 6:75,v=geometry.point ? 7:75;
    if (!mul(63,61,u) || !mul(68,62,v) || !mul(64,62,u) || !mul(69,61,v)) return false;
    if (!plusI) { negate(r[63]); negate(r[64]); }
    if (!plusJ) { negate(r[68]); negate(r[69]); }
    if (!sub(63,63,68) || !addAt(64,64,69)) return false;
    // Fifteen conversions: center3, arm2, rigid pair2, rate2, point2,
    // exact time1 and barycentrics3. Normal arm/point are source floats.
    const unsigned px=47+2*geometry.point,py=px+1;
    if (!narrow(40,67,i) || !narrow(41,67,j) || !mul(74,42,39) || !narrow(74,67,k)
        || !sub(73,px,40) || !narrow(73,67,3+i)
        || !sub(73,py,41) || !narrow(73,67,3+j)
        || !narrow(35,33,15) || !narrow(36,33,16)
        || !narrow(63,65,6+i) || !narrow(64,65,6+j)
        || !receipt(px,67,4+i) || !receipt(py,67,4+j)
        || !receipt(20,21,0)) return false;
    s[3+k]=-sigma*input.half[k]; s[6+k]=0;
    workspace.receipt[4+k]=input.plane;
    // Consume the candidates before overwriting47. Retain physical threshold
    // C_k(t)*d*S for the exact final-normal comparison in phase5.
    if (!mul(47,42,21) || !shift(47,47,g) || !sub(71,12,12)) return false;
    r[71].word[0]=1;
    if (!shift(71,71,g)) return false;
    for (unsigned a=0;a<3;++a) { s[9+a]=state.velocity[a]; s[12+a]=state.angular[a]; }
    const auto finite=[&](unsigned base) {
        return std::isfinite(s[base]) && std::isfinite(s[base+1]) && std::isfinite(s[base+2]);
    };
    if (!finite(0) || !finite(3) || !finite(6) || !std::isfinite(s[15]) || !std::isfinite(s[16])) return false;
    // Scratch vectors17/20/23/26/29 are disjoint three-float ranges. The
    // rounded rigid pair15/16 defines world-axis columns in source order.
    const auto dot=[&](unsigned a,unsigned b) {
        return s[a]*s[b]+s[a+1]*s[b+1]+s[a+2]*s[b+2];
    };
    const auto cross=[&](unsigned out,unsigned a,unsigned b) {
        s[out]=s[a+1]*s[b+2]-s[a+2]*s[b+1];
        s[out+1]=s[a+2]*s[b]-s[a]*s[b+2];
        s[out+2]=s[a]*s[b+1]-s[a+1]*s[b];
    };
    const auto axis=[&](unsigned column,unsigned row) {
        if (column==k) return row==k ? 1.0f:0.0f;
        if (row==k) return 0.0f;
        return column==i ? (row==i ? s[15]:s[16]):(row==i ? -s[16]:s[15]);
    };
    const float inverseMass=1/state.mass;
    if (!std::isfinite(inverseMass) || inverseMass<=0) return false;
    const auto inverseInertia=[&]() {
        s[20]=0; s[21]=0; s[22]=0;
        for (unsigned column=0;column<3;++column) {
            const float inertia=3*inverseMass/(input.half[(column+1)%3]*input.half[(column+1)%3]
                +input.half[(column+2)%3]*input.half[(column+2)%3]);
            const float component=inertia*(s[17]*axis(column,0)+s[18]*axis(column,1)+s[19]*axis(column,2));
            if (!std::isfinite(inertia) || inertia<=0 || !std::isfinite(component)) return false;
            for (unsigned row=0;row<3;++row) s[20+row]+=component*axis(column,row);
        }
        return finite(20);
    };
    const auto effectiveMass=[&](float& value) {
        cross(17,3,23);
        if (!finite(17) || !inverseInertia()) return false;
        cross(17,20,3); value=inverseMass+dot(17,23);
        return std::isfinite(value) && value>0;
    };
    const auto relativeVelocity=[&]() {
        for (unsigned a=0;a<3;++a) s[17+a]=s[12+a]-state.scheduledAngular[a];
        cross(20,17,3);
        for (unsigned a=0;a<3;++a) s[29+a]=s[9+a]+s[6+a]+s[20+a];
        return finite(29);
    };
    const auto apply=[&]() {
        cross(17,3,26);
        if (!finite(26) || !finite(17) || !inverseInertia()) return false;
        for (unsigned a=0;a<3;++a) {
            const float speed=s[26+a]*inverseMass,velocity=s[9+a]+speed,angular=s[12+a]+s[20+a];
            if (!std::isfinite(speed) || !std::isfinite(velocity) || !std::isfinite(angular)) return false;
            s[9+a]=std::clamp(velocity,-32.0f,32.0f);
            s[12+a]=std::clamp(angular,-32.0f,32.0f);
        }
        return true;
    };
    for (unsigned a=0;a<3;++a) s[23+a]=a==k ? float(sigma):0.0f;
    if (!relativeVelocity()) return false;
    const float closing=dot(29,23);
    if (!std::isfinite(closing)) return false;
    if (closing<0) {
        float mass;
        if (!effectiveMass(mass)) return false;
        const float normalImpulse=-(1+state.restitution)*closing/mass;
        if (!std::isfinite(normalImpulse) || normalImpulse<0) return false;
        for (unsigned a=0;a<3;++a) s[26+a]=s[23+a]*normalImpulse;
        if (!apply() || !relativeVelocity()) return false;
        const float afterNormal=dot(29,23);
        for (unsigned a=0;a<3;++a) s[23+a]=s[29+a]-afterNormal*(a==k ? float(sigma):0.0f);
        const float squared=dot(23,23),length=std::sqrt(squared);
        if (!std::isfinite(squared) || squared<0 || !std::isfinite(length)) return false;
        if (length>1.0e-7f) {
            for (unsigned a=0;a<3;++a) s[23+a]/=length;
            if (!effectiveMass(mass)) return false;
            const float cap=state.friction*normalImpulse,freeImpulse=length/mass;
            if (!std::isfinite(cap) || !std::isfinite(freeImpulse) || cap<0 || freeImpulse<0) return false;
            const float friction=std::min(cap,freeImpulse);
            for (unsigned a=0;a<3;++a) s[26+a]=-s[23+a]*friction;
            if (!apply()) return false;
        }
    }
    if (!budget.charge(ResponseCost::remainder(n))) return false;
    // Threshold q=sigma*C_k(t) overS. A rounded inward center never governs
    // projection. Only the exact42/71 pair selects the outward endpoint.
    if (!shift(48,42,0)) return false;
    if (sigma<0) negate(r[48]);
    bool above;
    float upper;
    if (!convertFloat(r[48],r[71],n,r[72],upper,above)
        || !supportedUpper(upper,above,upper)) return false;
    s[k]=sigma*upper;
    // Inspect every new operand even at root1, then perform six charged F.
    for (unsigned a=0;a<3;++a)
        if (!gridFits(s[a],g) || !gridFits(s[9+a],g)) return false;
    for (unsigned a=0;a<3;++a)
        if (!decodeFloat(r[49+a],s[a],n,g) || !decodeFloat(r[52+a],s[9+a],n,g)) return false;
    int comparison;
    if (!compare(r[49+k],r[42],n,comparison) || sigma*comparison<=0
        || !sub(56,21,20) || !shift(55,21,2*g)) return false;
    // For each axis: C*d then shiftS, dt*v, and that product*(d-n).
    // Exactly nine M. Physical threshold47 already has the same denominator55.
    for (unsigned a=0;a<3;++a) {
        if (!mul(57,49+a,21) || !shift(57,57,g) || !mul(58,10,52+a)
            || !mul(58,58,56) || !addAt(57,57,58)) return false;
        if (a==k && (!compare(r[57],r[47],n,comparison) || sigma*comparison<0)) return false;
        if (!narrow(57,55,a)) return false;
    }
    // One of the two paid G proofs checks the rounded stored endpoint.
    // The other covers projection; its decoded endpoint is already among the
    // six response values. Both exact and rounded remainder endpoints are safe.
    if (!gridFits(s[k],g) || !decodeFloat(r[58],s[k],n,g)
        || !compare(r[58],r[42],n,comparison) || sigma*comparison<0) return false;
    output.kind=Kind::impact; output.triangle=geometry.triangle; output.point=geometry.point;
    for (unsigned a=0;a<3;++a) {
        output.center[a]=s[a]; output.velocity[a]=s[9+a]; output.angular[a]=s[12+a];
        output.weights[a]=workspace.receipt[1+a]; output.pointOnTriangle[a]=workspace.receipt[4+a];
        output.impulse[a]=double(state.mass)*(double(s[9+a])-double(state.velocity[a]));
    }
    output.time=workspace.receipt[0];
    return true;
}
}
