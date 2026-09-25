#pragma once

#include "particle_geometry.h"
#include <limits>

namespace videorender
{
struct ParticleRigidBody
{
    float x=0,y=0,vx=0,vy=0,tx=0,ty=0,hue=0;
    int noteRow=-1;
    std::int64_t identity=0;
    std::uint64_t geometryIdentity=0;
    float z=0,vz=0,tz=0;
    std::array<float,4> orientation {0,0,0,1};
    std::array<float,3> angular {};
};

struct ParticleSolidState
{
    std::array<ParticleRigidBody,64> bodies {};
    int count=0;
    ParticleGeometryFrame colliders;
};

using SolidVector=std::array<float,3>;
inline float solidDot(const SolidVector& a,const SolidVector& b)
{ return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline SolidVector solidCross(const SolidVector& a,const SolidVector& b)
{ return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline SolidVector solidPosition(const ParticleRigidBody& b) { return {b.x,b.y,b.z}; }
inline SolidVector solidVelocity(const ParticleRigidBody& b) { return {b.vx,b.vy,b.vz}; }
inline void solidMove(ParticleRigidBody& b,const SolidVector& n,float distance)
{ b.x+=n[0]*distance; b.y+=n[1]*distance; b.z+=n[2]*distance; }
inline void solidVelocityAdd(ParticleRigidBody& b,const SolidVector& delta)
{
    b.vx=std::clamp(b.vx+delta[0],-32.0f,32.0f);
    b.vy=std::clamp(b.vy+delta[1],-32.0f,32.0f);
    b.vz=std::clamp(b.vz+delta[2],-32.0f,32.0f);
}

inline void solidRotate(ParticleRigidBody& b,float dt,float drag)
{
    auto& q=b.orientation;
    const auto old=q;
    for (auto& w:b.angular) w=std::clamp(w/(1+drag*dt),-32.0f,32.0f);
    const auto& w=b.angular;
    q[0]+=0.5f*dt*(w[0]*old[3]+w[1]*old[2]-w[2]*old[1]);
    q[1]+=0.5f*dt*(-w[0]*old[2]+w[1]*old[3]+w[2]*old[0]);
    q[2]+=0.5f*dt*(w[0]*old[1]-w[1]*old[0]+w[2]*old[3]);
    q[3]-=0.5f*dt*(w[0]*old[0]+w[1]*old[1]+w[2]*old[2]);
    const auto length=std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    for (auto& value:q) value/=length;
}

// Sphere impulses include tangential contact velocity and angular inertia.
// A null second body is a kinematic Geometry or unit-boundary contact.
inline void solidImpulse(ParticleRigidBody& a,ParticleRigidBody* b,const SolidVector& normal,
                         const SolidVector& surface,const ParticleParams& p)
{
    const float inverseMass=1/p.bodyMass,r=p.bodyRadius;
    SolidVector armA {-r*normal[0],-r*normal[1],-r*normal[2]};
    SolidVector armB {r*normal[0],r*normal[1],r*normal[2]};
    auto va=solidVelocity(a),vb=b ? solidVelocity(*b):surface;
    const auto spinA=solidCross(a.angular,armA),spinB=b ? solidCross(b->angular,armB):SolidVector{};
    SolidVector relative;
    for (int axis=0;axis<3;++axis) relative[axis]=va[axis]+spinA[axis]-vb[axis]-spinB[axis];
    const float speed=solidDot(relative,normal);
    if (speed>=0) return;
    const float impulse=-(1+p.restitution)*speed/(inverseMass*(b ? 2:1));
    SolidVector tangent;
    for (int axis=0;axis<3;++axis) tangent[axis]=relative[axis]-speed*normal[axis];
    const float tangentLength=std::sqrt(solidDot(tangent,tangent));
    const float friction=std::min(p.friction*impulse,tangentLength/(inverseMass*(b ? 7:3.5f)));
    SolidVector push;
    for (int axis=0;axis<3;++axis)
        push[axis]=impulse*normal[axis]-(tangentLength>1.0e-7f ? friction*tangent[axis]/tangentLength:0);
    const float inverseInertia=2.5f*inverseMass/(r*r);
    const auto torqueA=solidCross(armA,push),torqueB=solidCross(armB,push);
    SolidVector acceleration;
    for (int axis=0;axis<3;++axis) {
        acceleration[axis]=push[axis]*inverseMass;
        a.angular[axis]=std::clamp(a.angular[axis]+torqueA[axis]*inverseInertia,-32.0f,32.0f);
        if (b) b->angular[axis]=std::clamp(b->angular[axis]-torqueB[axis]*inverseInertia,-32.0f,32.0f);
    }
    solidVelocityAdd(a,acceleration);
    if (b) { for (auto& v:acceleration) v=-v; solidVelocityAdd(*b,acceleration); }
}

inline void solidSphereContact(ParticleRigidBody& a,ParticleRigidBody* b,
                              const SolidVector& center,float otherRadius,
                              const SolidVector& surface,const ParticleParams& p)
{
    const auto position=solidPosition(a);
    SolidVector normal;
    for (int axis=0;axis<3;++axis) normal[axis]=position[axis]-center[axis];
    const float distance=std::sqrt(solidDot(normal,normal)),reach=p.bodyRadius+otherRadius;
    if (distance>=reach) return;
    if (distance>1.0e-7f) for (auto& value:normal) value/=distance;
    else normal={1,0,0};
    solidMove(a,normal,(reach-distance)*(b ? 0.5f:1));
    if (b) solidMove(*b,normal,-(reach-distance)*0.5f);
    solidImpulse(a,b,normal,surface,p);
}

inline void solidSphereSweep(ParticleRigidBody& a,ParticleRigidBody* b,
                             const SolidVector& fromA,const SolidVector& fromB,
                             const SolidVector& toB,float otherRadius,const SolidVector& surface,
                             const ParticleParams& p)
{
    const auto toA=solidPosition(a);
    SolidVector origin,travel;
    for (int axis=0;axis<3;++axis) {
        origin[axis]=fromA[axis]-fromB[axis];
        travel[axis]=toA[axis]-fromA[axis]-(toB[axis]-fromB[axis]);
    }
    const float radius=p.bodyRadius+otherRadius;
    const float aa=solidDot(travel,travel),bb=2*solidDot(origin,travel),cc=solidDot(origin,origin)-radius*radius;
    const float discriminant=bb*bb-4*aa*cc;
    if (cc<0 || aa<1.0e-12f || discriminant<0) return;
    const float hit=(-bb-std::sqrt(discriminant))/(2*aa);
    if (hit<0 || hit>1) return;
    SolidVector normal,positionA,positionB;
    for (int axis=0;axis<3;++axis) {
        normal[axis]=(origin[axis]+hit*travel[axis])/radius;
        positionB[axis]=b ? fromB[axis]+hit*(toB[axis]-fromB[axis]):toB[axis];
        positionA[axis]=positionB[axis]+normal[axis]*radius;
    }
    a.x=positionA[0]; a.y=positionA[1]; a.z=positionA[2];
    if (b) { b->x=positionB[0]; b->y=positionB[1]; b->z=positionB[2]; }
    solidImpulse(a,b,normal,surface,p);
}

inline void solidGeometryContact(ParticleRigidBody& body,const ParticleRigidBody& previous,
                                 const ParticleGeometryFrame& frame,const ParticleGeometryFrame& old,
                                 float dt,const ParticleParams& p,bool sweep=true)
{
    if (frame.meshIdentity!=0) {
        auto position=solidPosition(body),from=solidPosition(previous);
        SolidVector low,high,surface,travel;
        bool inside=true;
        for (int axis=0;axis<3;++axis) {
            low[axis]=frame.minimum[axis]-p.bodyRadius; high[axis]=frame.maximum[axis]+p.bodyRadius;
            surface[axis]=std::clamp(((frame.minimum[axis]-old.minimum[axis])
                +(frame.maximum[axis]-old.maximum[axis]))*0.5f/dt,-32.0f,32.0f);
            from[axis]+=surface[axis]*dt; travel[axis]=position[axis]-from[axis];
            inside=inside && position[axis]>=low[axis] && position[axis]<=high[axis];
        }
        // Swept expanded AABB: conservative at corners, but cannot tunnel
        // through an admitted moving solid between fixed steps.
        float enter=0,leave=1; int hitAxis=-1; float sign=0;
        bool intersects=true;
        for (int axis=0;axis<3;++axis) {
            if (std::abs(travel[axis])<1.0e-9f) {
                if (from[axis]<low[axis] || from[axis]>high[axis]) intersects=false;
                continue;
            }
            float a=(low[axis]-from[axis])/travel[axis],b=(high[axis]-from[axis])/travel[axis];
            if (a>b) std::swap(a,b);
            if (a>=enter) { enter=a; hitAxis=axis; sign=travel[axis]>0 ? -1:1; }
            leave=std::min(leave,b);
        }
        SolidVector normal {};
        if (sweep && intersects && hitAxis>=0 && enter<=leave && enter<=1 && leave>=0) {
            normal[hitAxis]=sign;
            position[hitAxis]=sign<0 ? low[hitAxis]:high[hitAxis];
            body.x=position[0]; body.y=position[1]; body.z=position[2];
            solidImpulse(body,nullptr,normal,surface,p);
        } else if (inside) {
            float nearest=std::numeric_limits<float>::max();
            for (int axis=0;axis<3;++axis) for (int side=0;side<2;++side) {
                const float distance=side ? high[axis]-position[axis]:position[axis]-low[axis];
                if (distance<nearest) { nearest=distance; hitAxis=axis; sign=side ? 1:-1; }
            }
            normal[hitAxis]=sign; solidMove(body,normal,nearest);
            solidImpulse(body,nullptr,normal,surface,p);
        }
    }
    for (int i=0;i<frame.pointCount;++i) {
        SolidVector surface;
        for (int axis=0;axis<3;++axis) surface[axis]=std::clamp(
            (frame.solidPoints[i][axis]-old.solidPoints[i][axis])/dt,-32.0f,32.0f);
        if (sweep) solidSphereSweep(body,nullptr,solidPosition(previous),old.solidPoints[i],
            frame.solidPoints[i],p.colliderThickness,surface,p);
        solidSphereContact(body,nullptr,frame.solidPoints[i],p.colliderThickness,surface,p);
    }
    if (p.collisionMode!=0) for (int axis=0;axis<3;++axis) {
        const float value=solidPosition(body)[axis];
        if (value<p.bodyRadius || value>1-p.bodyRadius) {
            SolidVector normal {}; normal[axis]=value<p.bodyRadius ? 1:-1;
            solidMove(body,normal,value<p.bodyRadius ? p.bodyRadius-value:value-(1-p.bodyRadius));
            solidImpulse(body,nullptr,normal,{},p);
        }
    }
}
}
