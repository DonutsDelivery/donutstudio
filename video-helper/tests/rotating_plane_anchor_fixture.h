#pragma once

// Literal rotating-plane value anchors, shared by the GL visual plan executor
// test and the all-platform rotating-plane anchor test. The including file
// first declares check(bool,const char*) and an atomic allocations counter.
#include "../src/particle_rotating_plane_response.h"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <limits>

// The same checked charge rule as videorender::SolidTriangleWorkBudget,
// without the particle solver headers that need the GL executor build.
struct RotatingAnchorBudget
{
    std::uint64_t used=0,limit=100000000;
    bool charge(std::uint64_t count=1)
    { if (used>limit || count>limit-used) return false; used+=count; return true; }
};

namespace
{
// Pure value response fixtures. Their injected halves and raw coefficients do
// not establish an authenticated cube sample, capture or replay acceptance.
void checkRotatingPlaneResponsePrerequisite()
{
    namespace plane=videorender::rotatingplane;
    namespace exact=videorender::rotatingexact;
    static_assert(sizeof(plane::Workspace)==27768 && sizeof(plane::Response)==128);
    static_assert(plane::ResponseCost::rate(39)==4159982
        && plane::ResponseCost::remainder(39)==1120138 && plane::ResponseCost::guard(39)==946);
    plane::Workspace workspace;
    for (auto& value:workspace.integer) { value.word.fill(0xa5a5a5a5U); value.negative=0; }
    workspace.scalar.product.fill(0x5a5a5a5aU);
    plane::Input input{{0.25f,0.25f,0.0703125f},{0.25f,0.25f,0.0625f},
        {0.03125f,0.03125f,0.03125f},0.03125f,0.0078125f,0.0f,
        0.0f,1.0f,0.015625f,1.0f,1.0f,0.0f,
        exact::floatFromBits(0x3f7fe002U),exact::floatFromBits(0x3cfff000U),2};
    float positions[9]={0.1875f,exact::floatFromBits(0x2b800000U),0,
        0.265625f,0.1875f,0,0.1875f,0.265625f,0};
    std::uint32_t indices[6]={0,1,2,0,1,2};
    plane::Mesh mesh{positions,indices,3,1};
    plane::ResponseInput state{{0,0,-1},{0,0,4},{0,0,4},1,1,0.5f};
    plane::Response result;
    std::memset(&result,0xa5,sizeof(result));
    const auto unchanged=[&] {
        const auto* bytes=reinterpret_cast<const unsigned char*>(&result);
        for (unsigned a=0;a<sizeof(result);++a) if (bytes[a]!=0xa5) return false;
        return true;
    };
    const auto reset=[&] { std::memset(&result,0xa5,sizeof(result)); };
    const auto doubleBits=[](double value) {
        std::uint64_t bits; std::memcpy(&bits,&value,8); return bits;
    };
    RotatingAnchorBudget budget;
    const auto before=allocations.load(std::memory_order_relaxed);
    check(plane::response(input,mesh,state,39,workspace,budget,result)
        && result.kind==plane::Kind::impact && result.triangle==0 && result.point==1
        && budget.used==5440816,
        "literal endpoint value response completes both exact guards within the accepted phase charges");
    check(exact::floatBits(workspace.staging[6])==0x3e03e782U
        && exact::floatBits(workspace.staging[7])==0xbdf7d104U
        && workspace.staging[8]==0 && exact::floatBits(workspace.staging[15])==0x3f7fe002U
        && exact::floatBits(workspace.staging[16])==0x3cfff001U,
        "corner rate and rounded rigid inertia axes match independent exact derivative literals");
    check(exact::floatBits(workspace.staging[3])==0xbcf7e082U
        && exact::floatBits(workspace.staging[4])==0xbd03efc1U
        && exact::floatBits(workspace.staging[5])==0xbd000000U,
        "torque uses the independently rounded actual proxy arm");
    check(exact::floatBits(result.velocity[0])==0x3e0e3978U
        && exact::floatBits(result.velocity[1])==0x3e54db71U
        && exact::floatBits(result.velocity[2])==0xbf000000U
        && exact::floatBits(result.angular[0])==0xc16c2aafU
        && exact::floatBits(result.angular[1])==0x418492d5U
        && exact::floatBits(result.angular[2])==0x3f9af8e6U,
        "staged normal and binding Coulomb impulses match independent per-scalar binary32 anchors");
    check(exact::floatBits(result.center[0])==0x3e800000U
        && exact::floatBits(result.center[1])==0x3e800000U
        && exact::floatBits(result.center[2])==0x3d800001U && result.time==1,
        "endpoint response projects strictly outward and has exact zero remainder");
    check(doubleBits(result.weights[0])==0x3fae0f100000712aULL
        && doubleBits(result.weights[1])==0x3fda698c9999999aULL
        && doubleBits(result.weights[2])==0x3fe0ea48b3332c21ULL
        && doubleBits(result.pointOnTriangle[0])==0x3fcc207df8000000ULL
        && doubleBits(result.pointOnTriangle[1])==0x3fcbe081f8000000ULL
        && result.pointOnTriangle[2]==0 && result.impulse[2]==0.5,
        "receipt time, point and barycentrics round their exact proof factors independently");
    for (const auto& value:workspace.integer) for (unsigned word=39;word<87;++word)
        check(value.word[word]==0xa5a5a5a5U,"response reuses active words without increasing the wide register peak");
    for (unsigned word=78;word<174;++word)
        check(workspace.scalar.product[word]==0x5a5a5a5aU,"response preserves inactive product scratch words");
    mesh.triangles=2; budget={};
    check(plane::response(input,mesh,state,39,workspace,budget,result)
        && result.triangle==0 && result.point==1 && budget.used==5516600,
        "complete response retains exact duplicate source and point ties");
    mesh.triangles=1; indices[0]=2; indices[1]=1; indices[2]=0; budget={};
    check(plane::response(input,mesh,state,39,workspace,budget,result)
        && exact::floatBits(result.velocity[0])==0x3e0e3978U && result.point==1,
        "reversed authored winding preserves the actual response");
    indices[0]=0; indices[1]=1; indices[2]=2;
    // Cyclic basis and signed normal repeat the literal rate-aware response.
    for (unsigned k=0;k<3;++k) for (int sigma:{-1,1}) {
        const unsigned i=(k+1)%3,j=(k+2)%3;
        auto cyclic=input; auto cyclicState=state;
        cyclic.normalAxis=k;
        cyclic.previousCenter[i]=cyclic.previousCenter[j]=0.25f;
        cyclic.scheduledCenter[i]=cyclic.scheduledCenter[j]=0.25f;
        cyclic.previousCenter[k]=sigma*0.0703125f; cyclic.scheduledCenter[k]=sigma*0.0625f;
        cyclicState.velocity={}; cyclicState.angular={}; cyclicState.scheduledAngular={};
        cyclicState.velocity[k]=-sigma; cyclicState.angular[k]=cyclicState.scheduledAngular[k]=4;
        float cyclicPositions[9];
        for (unsigned vertex=0;vertex<3;++vertex) {
            cyclicPositions[3*vertex+i]=positions[3*vertex];
            cyclicPositions[3*vertex+j]=positions[3*vertex+1]; cyclicPositions[3*vertex+k]=0;
        }
        plane::Mesh cyclicMesh{cyclicPositions,indices,3,1}; budget={};
        check(plane::response(cyclic,cyclicMesh,cyclicState,39,workspace,budget,result)
            && exact::floatBits(result.velocity[i])==0x3e0e3978U
            && exact::floatBits(result.velocity[j])==0x3e54db71U
            && result.velocity[k]==-sigma*0.5f && result.center[k]==sigma*exact::floatFromBits(0x3d800001U),
            "cyclic y uses z,x and both sides preserve the literal corner impulse");
    }
    auto midpoint=input; midpoint.reach=exact::floatFromBits(0x3d000001U); budget={};
    check(plane::response(midpoint,mesh,state,39,workspace,budget,result)
        && exact::floatBits(result.center[2])==0x3d800001U && result.time==2097151.0/2097152.0,
        "exact normal midpoint tie selects the first supported upper float and retains its safe small remainder");
    // Actual binary32 source-fit dt and raw-pair boundary, still value-only.
    auto quarter=input; auto quarterState=state;
    quarter.previousCenter[2]=0.125f; quarter.scheduledCenter[2]=-0.125f;
    quarter.dt=exact::floatFromBits(0x3c088889U); quarter.rawZ=exact::floatFromBits(0x3c888889U);
    quarter.scheduledA=exact::floatFromBits(0x3f7fdb9aU); quarter.scheduledB=exact::floatFromBits(0x3d087ed5U);
    quarterState.velocity[2]=-30; quarterState.friction=0;
    reset(); budget={};
    check(!plane::response(quarter,mesh,quarterState,39,workspace,budget,result)
        && unchanged() && budget.used==5440816 && workspace.staging[11]==-15,
        "quarter cube corner keeps impulse15 private and refuses the literal negative exact remainder");
    float centeredPositions[9]={0,0,0,1,0,0,0,1,0};
    plane::Mesh centeredMesh{centeredPositions,indices,3,1}; budget={};
    check(plane::response(quarter,centeredMesh,quarterState,39,workspace,budget,result)
        && result.point==0 && result.time==0.25 && result.velocity[2]==30
        && exact::floatBits(result.center[2])==0x3e800001U
        && workspace.staging[6]==0 && workspace.staging[7]==0,
        "source-fit quarter face center has zero geometric rate and outward literal remainder536870949/2147483648");
    // Independent old/new radius-times-stored-ratio bits, with no sampler claim.
    const float ratio=exact::floatFromBits(0x3f13cd3aU);
    check(exact::floatBits(exact::floatFromBits(0x3d5db3d8U)*ratio)==0x3d000000U
        && exact::floatBits(exact::floatFromBits(0x3d5db3d7U)*ratio)==0x3cffffffU,
        "literal source-fit radius successor produces the required half and preserves the separated predecessor");
    auto oldHalf=input; oldHalf.half.fill(exact::floatFromBits(0x3cffffffU)); budget={};
    check(plane::response(oldHalf,mesh,state,39,workspace,budget,result)
        && result.kind==plane::Kind::separated && budget.used==107090,
        "superseded source-fit half is separated and receives no response phase");
    for (unsigned phase=0;phase<2;++phase) {
        reset(); budget={}; budget.limit=phase==0 ? 4320677:5440815;
        check(!plane::response(input,mesh,state,39,workspace,budget,result) && unchanged()
            && budget.used==(phase==0 ? 160696U:4320678U),
            "rate and remainder phase minus one refuse before staging with earlier work retained");
    }
    auto badState=state; badState.mass=exact::floatFromBits(0x00800000U); reset(); budget={};
    check(!plane::response(input,mesh,badState,39,workspace,budget,result)
        && unchanged() && budget.used==4320678,
        "finite source mass with overflowing staged inverse inertia refuses atomically");
    badState=state; badState.velocity[0]=std::numeric_limits<float>::infinity(); budget={};
    check(!plane::response(input,mesh,badState,39,workspace,budget,result) && unchanged(),
        "nonfinite incident response input cannot publish staged output");
    // Endpoint1 still inspects all six new operands before remainder work.
    badState=state; badState.velocity[0]=exact::floatFromBits(0x00800000U); badState.friction=0; budget={};
    check(!plane::response(input,mesh,badState,39,workspace,budget,result)
        && unchanged() && budget.used==5440816,
        "new velocity precision beyond g40 refuses even with exact zero remainder");
    float upper=17;
    check(plane::supportedUpper(0,false,upper) && exact::floatBits(upper)==0x00800000U
        && plane::supportedUpper(exact::floatFromBits(0x80000001U),false,upper) && upper==0
        && plane::supportedUpper(exact::floatFromBits(0x80800000U),false,upper) && upper==0
        && plane::supportedUpper(exact::floatFromBits(0x00800000U),false,upper)
        && exact::floatBits(upper)==0x00800001U,
        "supported projection skips the complete subnormal band and advances a min-normal tie");
    upper=17;
    check(!plane::supportedUpper(exact::floatFromBits(0x7f7fffffU),false,upper) && upper==17
        && plane::supportedUpper(exact::floatFromBits(0x3d800001U),true,upper)
        && exact::floatBits(upper)==0x3d800001U,
        "projection refuses outward FLT_MAX and retains an already strictly upper rounded value");
    auto zeroThreshold=input;
    zeroThreshold.plane=-0.0625f; zeroThreshold.previousCenter[2]=0.0078125f;
    zeroThreshold.scheduledCenter[2]=0;
    float shiftedPositions[9];
    for (unsigned a=0;a<9;++a) shiftedPositions[a]=positions[a];
    shiftedPositions[2]=shiftedPositions[5]=shiftedPositions[8]=-0.0625f;
    plane::Mesh shiftedMesh{shiftedPositions,indices,3,1}; reset(); budget={};
    check(!plane::response(zeroThreshold,shiftedMesh,state,39,workspace,budget,result) && unchanged(),
        "threshold zero refuses a fixed grid unable to decode positive min-normal");
    budget={};
    check(plane::response(zeroThreshold,shiftedMesh,state,77,workspace,budget,result)
        && exact::floatBits(result.center[2])==0x00800000U,
        "threshold zero accepts exactly positive min-normal at the admitted g127 width");
    reset(); budget={};
    auto* aliased=reinterpret_cast<plane::Response*>(&workspace);
    check(!plane::response(input,mesh,state,39,workspace,budget,*aliased) && budget.used==0 && unchanged(),
        "output aliasing the borrowed workspace refuses before any scratch write");
    check(allocations.load(std::memory_order_relaxed)==before,
        "all pure value response and projection fixtures use no heap allocation");
}

// These are mathematical coefficient fixtures. They never mint the real
// solidRotate capture or a body/replay/source authority, and execute no response.
void checkRotatingPlaneGeometricPrerequisite()
{
    namespace plane=videorender::rotatingplane;
    namespace exact=videorender::rotatingexact;
    static_assert(sizeof(plane::Workspace)==27768 && sizeof(plane::Result)==32);
    static_assert(plane::Cost::grid(22)==1 && plane::Cost::grid(39)==40 && plane::Cost::grid(87)==149);
    static_assert(plane::Cost::first(39,3,1)==31309 && plane::Cost::face(39)==53606
        && plane::Cost::triangle(39)==75781);
    plane::Workspace workspace;
    // Only tests initialize all storage. Production owns admission and passes
    // trivial borrowed storage without a full-width constructor or copy.
    for (auto& value:workspace.integer) {
        value.word.fill(0xa5a5a5a5U); value.negative=0xaaaaaaaaU;
    }
    workspace.scalar.product.fill(0x5a5a5a5aU);
    workspace.staging.fill(-19.0f); workspace.control.fill(-19); workspace.receipt.fill(-19.0);
    plane::Input input{{0.25f,0.25f,0.0703125f},{0.25f,0.25f,0.0625f},
        {0.03125f,0.03125f,0.03125f},0.03125f,0.0078125f,0.0f,
        0.0f,1.0f,0.015625f,1.0f,1.0f,0.0f,
        exact::floatFromBits(0x3f7fe002U),exact::floatFromBits(0x3cfff000U),2};
    float positions[9]={0.1875f,exact::floatFromBits(0x2b800000U),0.0f,
        0.265625f,0.1875f,0.0f,0.1875f,0.265625f,0.0f};
    std::uint32_t indices[6]={0,1,2,0,1,2};
    plane::Mesh mesh{positions,indices,3,1};
    plane::Result result{plane::Kind::separated,7,8,9,-9,10,11,12};
    const auto unchanged=[&] { return result.kind==plane::Kind::separated && result.limbs==7
        && result.grid==8 && result.normalAxis==9 && result.side==-9
        && result.triangle==10 && result.point==11 && result.reserved==12; };
    const auto resetResult=[&] { result={plane::Kind::separated,7,8,9,-9,10,11,12}; };
    RotatingAnchorBudget budget;
    const auto before=allocations.load(std::memory_order_relaxed);
    check(plane::certificate(input,mesh,39,workspace,budget,result)
        && result.kind==plane::Kind::impact && result.grid==40 && result.side==1
        && result.triangle==0 && result.point==1 && budget.used==160696,
        "literal endpoint rotating geometry selects the minus/minus finite corner at its exact root");
    // Compare to independent literal fractions by cross multiplication. This
    // checker contains no quaternion, root, face or determinant construction.
    const auto rational=[&](const exact::Integer& numerator,const exact::Integer& denominator,
                            std::uint64_t expectedNumerator,std::uint64_t expectedDenominator,
                            unsigned denominatorShift=0) {
        exact::Integer a {},b {},left {},right {};
        exact::Workspace scratch;
        a.word[0]=std::uint32_t(expectedNumerator); a.word[1]=std::uint32_t(expectedNumerator>>32);
        b.word[0]=std::uint32_t(expectedDenominator); b.word[1]=std::uint32_t(expectedDenominator>>32);
        int comparison=17;
        return exact::shiftLeft(b,b,39,denominatorShift)
            && exact::multiply(left,numerator,b,39,scratch)
            && exact::multiply(right,denominator,a,39,scratch)
            && exact::compare(left,right,39,comparison) && comparison==0;
    };
    auto& r=workspace.integer;
    check(rational(r[20],r[21],1,1),"the endpoint certificate retains exact root one");
    check(rational(r[49],r[67],58986431,1,28) && rational(r[50],r[67],58462271,1,28),
        "selected corner XY matches the independent exact endpoint point");
    // Selected signs use unscaled triangle edges and common denominator S*S*D.
    // Test-only scratch constructs it, without a geometry algorithm oracle.
    exact::Integer edgeDenominator {};
    check(exact::shiftLeft(edgeDenominator,r[39],39,80)
        && rational(r[63],edgeDenominator,3237614362814582719ULL,1,68)
        && rational(r[64],edgeDenominator,10465285,1,33)
        && rational(r[65],edgeDenominator,2527696675841765441ULL,1,68),
        "all three endpoint oriented signs match independent literal fractions");
    check(rational(r[37],r[39],8384513,8388608)
        && rational(r[38],r[39],4095,131072),
        "the impact endpoint axes retain the released float hull exactly");
    for (const auto& value:r) for (unsigned word=39;word<87;++word)
        check(value.word[word]==0xa5a5a5a5U,"geometric phases preserve every inactive register word");
    for (unsigned word=78;word<174;++word)
        check(workspace.scalar.product[word]==0x5a5a5a5aU,"geometric products preserve inactive full-product words");
    check(workspace.receipt[0]==-19.0 && workspace.staging[20]==-19.0f,
        "geometric prefix leaves response staging and receipt conversions untouched");

    // Every triangle is classified. Duplicates use first source order; a later
    // reversed witness cannot replace the earlier exact point tie.
    mesh.triangles=2; budget={};
    check(plane::certificate(input,mesh,39,workspace,budget,result)
        && result.triangle==0 && result.point==1 && budget.used==236480,
        "exact duplicate triangle ties retain the first source and first point");
    indices[3]=2; indices[4]=1; indices[5]=0; budget={};
    check(plane::certificate(input,mesh,39,workspace,budget,result) && result.triangle==0 && result.point==1,
        "a later reversed triangle winding preserves the first selected witness");
    mesh.triangles=1;
    indices[0]=2; indices[1]=1; indices[2]=0; budget={};
    check(plane::certificate(input,mesh,39,workspace,budget,result) && result.point==1
        && rational(r[63],edgeDenominator,10465285,1,33)
        && rational(r[64],edgeDenominator,3237614362814582719ULL,1,68)
        && rational(r[65],edgeDenominator,2527696675841765441ULL,1,68),
        "reversed winding normalizes exact edge signs without changing containment");
    indices[0]=0; indices[1]=1; indices[2]=2;

    // The same independent XY anchors survive the right-handed cyclic basis
    // and both contacting faces. This is z,x for a y-normal plane.
    for (unsigned k=0;k<3;++k) for (int sigma:{-1,1}) {
        auto cyclic=input;
        const unsigned i=(k+1)%3,j=(k+2)%3;
        cyclic.normalAxis=k;
        cyclic.previousCenter[i]=0.25f; cyclic.previousCenter[j]=0.25f;
        cyclic.scheduledCenter[i]=0.25f; cyclic.scheduledCenter[j]=0.25f;
        cyclic.previousCenter[k]=sigma*0.0703125f; cyclic.scheduledCenter[k]=sigma*0.0625f;
        float cyclicPositions[9];
        for (unsigned vertex=0;vertex<3;++vertex) {
            cyclicPositions[3*vertex+i]=positions[3*vertex];
            cyclicPositions[3*vertex+j]=positions[3*vertex+1];
            cyclicPositions[3*vertex+k]=0.0f;
        }
        plane::Mesh cyclicMesh{cyclicPositions,indices,3,1}; budget={};
        check(plane::certificate(cyclic,cyclicMesh,39,workspace,budget,result)
            && result.normalAxis==k && result.side==sigma && result.point==1
            && rational(r[49],r[67],58986431,1,28) && rational(r[50],r[67],58462271,1,28),
            "cyclic coordinate planes and both sides preserve the literal finite corner");
    }

    float largeTriangle[9]={0,0,0,1,0,0,0,1,0};
    plane::Mesh centeredMesh{largeTriangle,indices,3,1};
    auto quarter=input; quarter.previousCenter[2]=0.125f; quarter.scheduledCenter[2]=-0.125f;
    budget={};
    check(plane::certificate(quarter,centeredMesh,39,workspace,budget,result)
        && result.point==0 && rational(r[20],r[21],1,4)
        && rational(r[37],r[39],9009260805554177ULL,9009535750504448ULL)
        && rational(r[38],r[39],1099779997695ULL,140773996101632ULL)
        && rational(r[47],r[67],1,4) && rational(r[48],r[67],1,4),
        "quarter root retains the exact normalized raw lift and first center witness");
    float vertexTriangle[9]={0.25f,0.25f,0,1,0.25f,0,0.25f,1,0};
    plane::Mesh vertexMesh{vertexTriangle,indices,3,1}; budget={};
    check(plane::certificate(input,vertexMesh,39,workspace,budget,result) && result.point==0
        && rational(r[63],r[66],0,1) && rational(r[64],r[66],1,1) && rational(r[65],r[66],0,1),
        "closed containment admits an exact triangle vertex with literal barycentrics");
    auto zero=input; zero.previousCenter[2]=0.0625f; zero.scheduledCenter[2]=0.03125f; budget={};
    check(plane::certificate(zero,centeredMesh,39,workspace,budget,result)
        && result.point==0 && rational(r[20],r[21],0,1),"time-zero approach has an exact zero first root");
    zero.scheduledCenter[2]=0.125f; budget={};
    check(plane::certificate(zero,centeredMesh,39,workspace,budget,result)
        && result.kind==plane::Kind::separated && budget.used==107090,
        "exact time-zero strict departure is separated without a triangle event");
    zero.scheduledCenter[2]=0.0625f; budget={}; resetResult();
    check(!plane::certificate(zero,centeredMesh,39,workspace,budget,result) && unchanged(),
        "persistent tangency refuses without publishing mathematical outputs");
    zero.previousCenter[2]=0.03125f; budget={};
    check(!plane::certificate(zero,centeredMesh,39,workspace,budget,result) && unchanged(),
        "initial penetration refuses without a first-root claim");
    auto oldHalf=input; oldHalf.half.fill(exact::floatFromBits(0x3cffffffU)); budget={};
    check(plane::certificate(oldHalf,mesh,39,workspace,budget,result)
        && result.kind==plane::Kind::separated,"the superseded literal half is strictly separated at the endpoint");
    auto zeroReach=input; zeroReach.reach=0; zeroReach.scheduledCenter[2]=0.03125f; budget={};
    check(plane::certificate(zeroReach,centeredMesh,39,workspace,budget,result)
        && rational(r[20],r[21],1,1),"zero thickness retains an exact finite-triangle endpoint witness");

    float emptyTriangle[9]={2,2,0,3,2,0,2,3,0};
    plane::Mesh emptyMesh{emptyTriangle,indices,3,1}; budget={}; resetResult();
    check(!plane::certificate(input,emptyMesh,39,workspace,budget,result) && unchanged()
        && budget.used==160696,"all root triangles disjoint refuses an unsolved later finite contact");
    float mixedPositions[18]={0.25f,0.25390625f,0,0.25390625f,0.25390625f,0,0.25f,0.2578125f,0,
        0.1875f,exact::floatFromBits(0x2b800000U),0,0.265625f,0.1875f,0,0.1875f,0.265625f,0};
    std::uint32_t mixedIndices[6]={0,1,2,3,4,5};
    plane::Mesh mixedMesh{mixedPositions,mixedIndices,6,2}; budget={};
    check(!plane::certificate(input,mixedMesh,39,workspace,budget,result) && unchanged(),
        "a triangle inside the face before a later witness remains unclassified and refuses");
    mixedIndices[0]=3; mixedIndices[1]=4; mixedIndices[2]=5;
    mixedIndices[3]=0; mixedIndices[4]=1; mixedIndices[5]=2; budget={};
    check(!plane::certificate(input,mixedMesh,39,workspace,budget,result) && unchanged(),
        "an unknown later triangle refuses even after finding an earlier finite witness");

    // An edge-crossing triangle also intersects the face without containing
    // any of the five candidates. A later real witness must not hide it.
    mixedPositions[0]=0.125f; mixedPositions[1]=0.25390625f;
    mixedPositions[3]=0.375f; mixedPositions[4]=0.25390625f;
    mixedPositions[6]=0.375f; mixedPositions[7]=0.26171875f;
    mixedIndices[0]=0; mixedIndices[1]=1; mixedIndices[2]=2;
    mixedIndices[3]=3; mixedIndices[4]=4; mixedIndices[5]=5; budget={};
    check(!plane::certificate(input,mixedMesh,39,workspace,budget,result) && unchanged(),
        "an unconstructed edge crossing before a later witness refuses transactionally");
    for (unsigned coordinate=0;coordinate<9;++coordinate) mixedPositions[coordinate]=emptyTriangle[coordinate];
    budget={};
    check(plane::certificate(input,mixedMesh,39,workspace,budget,result)
        && result.triangle==1 && result.point==1,
        "an exactly pruned earlier triangle permits the next source-ordered finite witness");

    // Literal phase-minus-one checks retain completed attempt charges.
    for (unsigned phase=0;phase<3;++phase) {
        budget={}; budget.limit=phase==0 ? 31308:(phase==1 ? 84914:160695); resetResult();
        check(!plane::certificate(input,mesh,39,workspace,budget,result) && unchanged()
            && budget.used==(phase==0 ? 0U:(phase==1 ? 31309U:84915U)),
            "each exact geometric phase is admitted atomically before its operations");
    }
    budget={}; resetResult();
    check(!plane::certificate(input,mesh,38,workspace,budget,result) && unchanged(),
        "frame exponent forty refuses the unchanged smaller fixed grid");
    auto invalid=input; invalid.scheduledA=0; budget={};
    check(!plane::certificate(invalid,mesh,39,workspace,budget,result) && unchanged(),
        "large endpoint deformation cannot pass the checked rounding residual bound");
    invalid=input; invalid.rawW=-1; budget={};
    check(!plane::certificate(invalid,mesh,39,workspace,budget,result) && unchanged(),
        "missing common-sign raw quaternion certificate refuses");
    const float previousNormal=positions[2]; positions[2]=0.125f; budget={};
    check(!plane::certificate(input,mesh,39,workspace,budget,result) && unchanged(),
        "noncoplanar authored triangle values refuse before exact geometry");
    positions[2]=previousNormal;
    indices[2]=3; budget={};
    check(!plane::certificate(input,mesh,39,workspace,budget,result) && unchanged(),
        "invalid source index refuses before triangle dereference");
    indices[2]=1; budget={};
    check(!plane::certificate(input,mesh,39,workspace,budget,result) && unchanged(),
        "exact degenerate triangle area refuses");
    budget={};
    auto separated=input; separated.scheduledCenter[2]=0.125f;
    check(!plane::certificate(separated,mesh,39,workspace,budget,result) && unchanged(),
        "whole-plane separation still refuses a degenerate source triangle");
    indices[2]=2; budget={};
    plane::Mesh aliasedMesh{workspace.staging.data(),indices,3,1};
    check(!plane::certificate(input,aliasedMesh,39,workspace,budget,result) && unchanged() && budget.used==31309,
        "borrowed mesh cannot alias mutable certificate staging");
    check(allocations.load(std::memory_order_relaxed)==before,
        "the borrowed exact geometry and literal rational checks allocate no heap storage");
}

void checkRotatingExactScalarPrerequisite()
{
    using namespace videorender::rotatingexact;
    static_assert(sizeof(Integer)==352 && sizeof(Workspace)==696);
    static_assert(Cost::decode(39)==51 && Cost::add(39)==125 && Cost::compare(39)==43
        && Cost::shift(39)==45 && Cost::multiply(39)==1728 && Cost::convert(39)==274576);
    Integer a {},b {},out {},remainder {};
    Workspace workspace;
    int order=17; bool above=false; float rounded=0; double wide=0;
    std::uint64_t wideBits=0;
    const auto doubleBits=[](double value) {
        std::uint64_t result; std::memcpy(&result,&value,sizeof(result)); return result;
    };
    const auto reset=[&](unsigned n) {
        check(clear(a,n) && clear(b,n) && clear(out,n) && clear(remainder,n),
            "scalar fixtures explicitly initialize only their active registers");
    };
    reset(2); out.word[2]=0xdeadbeefU;
    check(decodeFloat(a,-7,2,0) && a.word[0]==7 && a.negative==1
        && decodeFloat(b,3,2,0) && b.word[0]==3 && b.negative==0,
        "normal floats decode into literal signed integers on the selected grid");
    check(add(out,a,b,2) && out.word[0]==4 && out.word[1]==0 && out.negative==1
        && out.word[2]==0xdeadbeefU,"signed addition preserves inactive storage");
    check(subtract(out,a,b,2) && out.word[0]==10 && out.negative==1,
        "signed subtraction uses the effective operand sign");
    check(add(a,a,b,2) && a.word[0]==4 && a.negative==1,
        "addition permits a destination alias without copying its source");
    check(subtract(b,b,b,2) && b.word[0]==0 && b.word[1]==0 && b.negative==0,
        "an exact cancellation produces canonical zero");
    b.negative=1;
    check(compare(b,out,2,order) && order==1,"comparison recognizes signed zero as zero");
    check(compare(b,b,2,order) && order==0,"two zero magnitudes compare equal regardless of sign");
    check(decodeFloat(a,0.125f,2,3) && a.word[0]==1 && a.negative==0,
        "decode removes exact mantissa powers before grid admission");
    out.word[0]=123; out.negative=0;
    check(!decodeFloat(out,0.125f,2,2) && out.word[0]==123,
        "insufficient fractional grid refuses before a destination write");
    check(!decodeFloat(out,floatFromBits(1),2,149)
        && !decodeFloat(out,floatFromBits(0x7f800000),2,0)
        && !decodeFloat(out,floatFromBits(0x7fc00000),2,0),
        "source decode refuses subnormal infinity and NaN coefficients");
    check(decodeFloat(out,floatFromBits(0x80000000),2,0) && out.word[0]==0 && out.negative==0,
        "negative float zero decodes to canonical integer zero");
    check(decodeFloat(out,floatFromBits(0x00800000),1,126) && out.word[0]==1,
        "minimum normal float fits its exact admitted grid");
    reset(9);
    check(decodeFloat(out,floatFromBits(0x7f7fffff),4,0)
        && out.word[3]==0xffffff00U && out.word[2]==0,
        "maximum finite float has its independent literal integer encoding");
    check(!decodeFloat(out,floatFromBits(0x7f7fffff),8,149)
        && decodeFloat(out,floatFromBits(0x7f7fffff),9,149)
        && out.word[7]==0xe0000000U && out.word[8]==0x001fffffU,
        "decode checks leading width as well as fractional precision");
    reset(3); a.word[0]=7; a.negative=1;
    check(shiftLeft(a,a,3,33) && a.word[0]==0 && a.word[1]==14 && a.word[2]==0 && a.negative==1,
        "descending shift preserves an aliased signed source across words");
    check(!shiftLeft(out,a,3,64),"shift refuses a discarded nonzero high word");
    clear(a,3);
    check(shiftLeft(out,a,3,std::numeric_limits<unsigned>::max()) && out.word[0]==0 && out.negative==0,
        "a large zero shift remains bounded and produces canonical zero");
    reset(2); a.word[0]=0xffffffffU; b.word[0]=1;
    check(add(out,a,b,2) && out.word[0]==0 && out.word[1]==1 && out.negative==0,
        "same-sign addition carries into its next active word");
    a.word[0]=3; b.word[0]=5;
    check(subtract(b,a,b,2) && b.word[0]==2 && b.negative==1,
        "reverse subtraction permits the right source to alias the result");
    reset(2); a.word[0]=0xffffffffU; b.word[1]=1;
    check(compare(a,b,2,order) && order==-1,"the highest unequal word decides magnitude order");
    reset(3); a.word[0]=0xffffffffU; a.word[1]=1;
    check(shiftLeft(a,a,3,4) && a.word[0]==0xfffffff0U && a.word[1]==31 && a.word[2]==0,
        "partial shifts join adjacent source words exactly");
    reset(1); a.word[0]=0xffffffffU; b.word[0]=1;
    check(!add(out,a,b,1),"addition reports the active-width carry overflow");
    a.word[0]=1;
    check(shiftLeft(a,a,1,31) && a.word[0]==0x80000000U
        && !shiftLeft(out,a,1,1),"the exact shift width boundary is checked");
    reset(2); a.word[0]=b.word[0]=0xffffffffU; a.negative=1;
    out.word[2]=0xdeadbeefU; workspace.product[4]=0xcafebabeU;
    check(multiply(out,a,b,2,workspace) && out.word[0]==1 && out.word[1]==0xfffffffeU
        && out.negative==1 && out.word[2]==0xdeadbeefU && workspace.product[4]==0xcafebabeU,
        "full carry multiplication matches its literal product and leaves inactive scratch alone");
    check(multiply(a,a,b,2,workspace) && a.word[0]==1 && a.word[1]==0xfffffffeU,
        "multiplication consumes its complete source before an aliased destination write");
    reset(1); a.word[0]=0xffffffffU; b.word[0]=2; out.word[0]=19;
    check(!multiply(out,a,b,1,workspace) && out.word[0]==19,
        "product overflow refuses before copying any result words");
    reset(87); a.word[86]=b.word[86]=0xffffffffU; out.word[0]=19;
    check(!multiply(out,a,b,87,workspace) && workspace.product[172]==1
        && workspace.product[173]==0xfffffffeU && out.word[0]==19,
        "the 174-word product contains and checks the maximum-width overflow");
    b.word[86]=0; b.word[0]=1;
    check(multiply(out,a,b,87,workspace) && out.word[86]==0xffffffffU && out.word[0]==0,
        "the largest active register remains usable below overflow");

    // Expected bits below are literal IEEE midpoint/sign/exponent oracles.
    // The fixture constructs exact integers; it does not recreate division.
    reset(2); a.word[0]=1; b.word[0]=3;
    const auto before=allocations.load(std::memory_order_relaxed);
    check(convertFloat(a,b,2,remainder,rounded,above) && floatBits(rounded)==0x3eaaaaabU && above,
        "one third rounds upward to the independently specified binary32 value");
    check(convertDouble(a,b,2,remainder,wide,above) && doubleBits(wide)==0x3fd5555555555555ULL && !above,
        "one third rounds downward to the independently specified binary64 value");
    a.negative=1;
    check(convertFloat(a,b,2,remainder,rounded,above) && floatBits(rounded)==0xbeaaaaabU && !above
        && convertDouble(a,b,2,remainder,wide,above) && doubleBits(wide)==0xbfd5555555555555ULL && above,
        "rounding direction is relative to the signed exact rational");
    a.negative=0; b.negative=1;
    check(convertDouble(a,b,2,remainder,wide,above) && doubleBits(wide)==0xbfd5555555555555ULL,
        "a signed denominator supplies the quotient sign");
    reset(1); a.word[0]=0xfffffffeU; b.word[0]=0xffffffffU;
    check(convertFloat(a,b,1,remainder,rounded,above) && floatBits(rounded)==0x3f800000U && above,
        "streamed subtraction handles a carried full-width remainder");
    reset(2); a.word[0]=16777217; b.word[0]=16777216;
    check(convertFloat(a,b,2,remainder,rounded,above) && floatBits(rounded)==0x3f800000U && !above,
        "the lower even float wins an exact midpoint tie");
    a.word[0]=16777219;
    check(convertFloat(a,b,2,remainder,rounded,above) && floatBits(rounded)==0x3f800002U && above,
        "the upper even float wins the adjacent midpoint tie");
    a.negative=1;
    check(convertFloat(a,b,2,remainder,rounded,above) && floatBits(rounded)==0xbf800002U && !above,
        "negative midpoint selection uses even magnitude and signed direction");
    reset(2); a.word[0]=1; a.word[1]=b.word[1]=0x00200000U;
    check(convertDouble(a,b,2,remainder,wide,above) && doubleBits(wide)==0x3ff0000000000000ULL && !above,
        "the binary64 lower even midpoint is exact");
    a.word[0]=3;
    check(convertDouble(a,b,2,remainder,wide,above) && doubleBits(wide)==0x3ff0000000000002ULL && above,
        "the binary64 upper even midpoint is exact");
    reset(1); a.word[0]=33554431; b.word[0]=33554432;
    check(convertFloat(a,b,1,remainder,rounded,above) && floatBits(rounded)==0x3f800000U && above,
        "rounding carry crosses a binade exactly");
    reset(5); a.word[0]=1; b.word[4]=0x00400000U;
    check(convertFloat(a,b,5,remainder,rounded,above) && floatBits(rounded)==0 && !above,
        "the half-minimum-subnormal float tie rounds to even zero");
    a.negative=1;
    check(convertFloat(a,b,5,remainder,rounded,above) && floatBits(rounded)==0x80000000U && above,
        "a negative underflow tie retains its IEEE zero sign");
    a.negative=0; a.word[0]=3;
    check(convertFloat(a,b,5,remainder,rounded,above) && floatBits(rounded)==2 && above,
        "the next subnormal midpoint chooses even significand two");
    a.word[0]=16777215;
    check(convertFloat(a,b,5,remainder,rounded,above) && floatBits(rounded)==0x00800000U && above,
        "the largest-subnormal midpoint promotes to minimum normal");
    reset(4); a.word[0]=1; b.word[3]=0x40000000U;
    check(convertFloat(a,b,4,remainder,rounded,above) && floatBits(rounded)==0x00800000U && !above,
        "minimum normal conversion is exact");
    reset(34); a.word[0]=1; b.word[33]=0x00080000U;
    check(convertDouble(a,b,34,remainder,wide,above) && doubleBits(wide)==0 && !above,
        "binary64 half-minimum-subnormal tie rounds to zero");
    a.word[0]=3;
    check(convertDouble(a,b,34,remainder,wide,above) && doubleBits(wide)==2 && above,
        "binary64 subnormal tie rounds to even significand two");
    a.word[0]=1; b.word[33]=0x00040000U;
    check(convertDouble(a,b,34,remainder,wide,above) && doubleBits(wide)==1 && !above,
        "binary64 minimum subnormal is exact");
    reset(32); a.word[0]=1; b.word[31]=0x40000000U;
    check(convertDouble(a,b,32,remainder,wide,above) && doubleBits(wide)==0x0010000000000000ULL,
        "binary64 minimum normal is exact");
    reset(4); a.word[3]=0xffffff00U; b.word[0]=1;
    check(convertFloat(a,b,4,remainder,rounded,above) && floatBits(rounded)==0x7f7fffffU && !above,
        "maximum finite float converts without a native floating operation");
    a.word[3]=0x80000080U;
    check(convertFloat(a,b,4,remainder,rounded,above) && floatBits(rounded)==0x7f000000U && !above,
        "large integer midpoint selects its lower even float");
    a.word[0]=1;
    check(convertFloat(a,b,4,remainder,rounded,above) && floatBits(rounded)==0x7f000001U && above,
        "unread low numerator bits make a non-tie even when division remainder is zero");
    clear(a,4); a.word[3]=0xffffff80U; rounded=1; above=true;
    check(!convertFloat(a,b,4,remainder,rounded,above) && floatBits(rounded)==0x3f800000U && above,
        "the float overflow midpoint refuses without publishing infinity or changing outputs");
    reset(32); a.word[31]=0xffffffffU; a.word[30]=0xfffff800U; b.word[0]=1;
    check(convertDouble(a,b,32,remainder,wide,above) && doubleBits(wide)==0x7fefffffffffffffULL && !above,
        "maximum finite double converts exactly");
    a.word[31]=0x80000000U; a.word[30]=0x00000400U;
    check(convertDouble(a,b,32,remainder,wide,above) && doubleBits(wide)==0x7fe0000000000000ULL,
        "large double midpoint selects its lower even result");
    a.word[0]=1;
    check(convertDouble(a,b,32,remainder,wide,above) && doubleBits(wide)==0x7fe0000000000001ULL && above,
        "unread numerator bits also affect binary64 sticky rounding");
    clear(a,32); a.word[31]=0xffffffffU; a.word[30]=0xfffffc00U; wide=1; above=true;
    check(!convertDouble(a,b,32,remainder,wide,above) && doubleBits(wide)==0x3ff0000000000000ULL && above,
        "the double overflow midpoint refuses without updating caller output");
    reset(87); a.word[0]=1; b.word[86]=0x80000000U;
    check(convertDouble(a,b,87,remainder,wide,above) && doubleBits(wide)==0,
        "the maximum admitted division width has a bounded underflow result");
    clear(a,87);
    check(convertBits(a,b,87,true,remainder,wideBits,above) && wideBits==0 && !above,
        "an exact zero quotient uses canonical zero");
    a.word[0]=1; clear(b,87); rounded=1; above=true;
    check(!convertFloat(a,b,87,remainder,rounded,above) && floatBits(rounded)==0x3f800000U && above,
        "zero denominator refuses before any converted publication");
    b.word[0]=1;
    check(!convertFloat(a,b,87,a,rounded,above) && a.word[0]==1,
        "remainder aliasing refuses before destroying an input");
    check(Cost::decode(0)==Cost::unavailable && Cost::convert(88)==Cost::unavailable
        && !decodeFloat(out,1,1,150) && !clear(out,0) && !clear(out,88) && !compare(a,b,0,order)
        && !convertFloat(a,b,88,remainder,rounded,above),
        "invalid active widths refuse before indexing storage");
    check(allocations.load(std::memory_order_relaxed)==before,
        "all exact arithmetic and rational conversions allocate no heap storage");
}
}
