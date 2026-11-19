#pragma once

#include "SdfIr.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <limits>
#include <new>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace videowire::sdfsurface {
inline constexpr std::uint32_t kAlgorithmVersion = 1;
// The wire counts records in 32 bits; the evaluator looks them up by id and has no record array to overrun.
inline constexpr std::size_t kMaximumRecords = 0xffffffffu;
inline constexpr std::uint64_t kMaximumStableId = 9007199254740991ull;
// Lattice samples address each other through 32-bit indices.
inline constexpr std::size_t kMaximumSamples = 0xffffffffu;
struct Request final {
  std::uint32_t algorithmVersion = kAlgorithmVersion;
  SdfIr geometry;
  std::array<double,3> minimum{-1.5,-1.5,-1.5}, maximum{1.5,1.5,1.5};
  std::array<std::uint32_t,3> cells{23,23,23};
  double gradientStep = 0.0001;
  double maximumResidual = 0.01;
};
inline bool equal(const Request& a,const Request& b) {
  if (a.algorithmVersion!=b.algorithmVersion || a.minimum!=b.minimum || a.maximum!=b.maximum
      || a.cells!=b.cells || a.gradientStep!=b.gradientStep || a.maximumResidual!=b.maximumResidual
      || a.geometry.schemaVersion!=b.geometry.schemaVersion || a.geometry.rootId!=b.geometry.rootId
      || a.geometry.records.size()!=b.geometry.records.size()
      || a.geometry.textOutlines.size()!=b.geometry.textOutlines.size()) return false;
  for (std::size_t i=0;i<a.geometry.records.size();++i) {
    const auto& x=a.geometry.records[i]; const auto& y=b.geometry.records[i];
    if (x.stableId!=y.stableId || x.operation!=y.operation || x.inputCount!=y.inputCount
        || x.inputs!=y.inputs || x.parameterCount!=y.parameterCount || x.parameters!=y.parameters
        || x.resourceStableId!=y.resourceStableId) return false;
  }
  for (std::size_t i=0;i<a.geometry.textOutlines.size();++i) {
    const auto& x=a.geometry.textOutlines[i]; const auto& y=b.geometry.textOutlines[i];
    if(x.stableId!=y.stableId || sdfTextOutlineIdentityBytes(x)!=sdfTextOutlineIdentityBytes(y)) return false;
  }
  return true;
}
namespace detail {
using Point = std::array<double,3>;
inline Point add(Point a,Point b) { for (unsigned i=0;i<3;++i) a[i]+=b[i]; return a; }
inline Point sub(Point a,Point b) { for (unsigned i=0;i<3;++i) a[i]-=b[i]; return a; }
inline Point mul(Point a,double b) { for (auto& x:a) x*=b; return a; }
inline double dot(Point a,Point b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
inline double length(Point a) { return std::sqrt(dot(a,a)); }
inline Point cross(Point a,Point b) { return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]}; }
inline double box(Point q,Point b) {
  for (unsigned i=0;i<3;++i) q[i]=std::abs(q[i])-b[i];
  return length({std::max(q[0],0.0),std::max(q[1],0.0),std::max(q[2],0.0)})
      +std::min(std::max({q[0],q[1],q[2]}),0.0);
}
inline double smooth(double a,double b,double k) {
  const auto h=std::clamp(0.5+0.5*(b-a)/k,0.0,1.0); return b+(a-b)*h-k*h*(1-h);
}
inline double modulo(double x,double period) { return x-period*std::floor(x/period+0.5); }
inline void rotatePair(double& x,double& y,double angle) {
  const auto old=x; x=std::cos(angle)*x-std::sin(angle)*y; y=std::sin(angle)*old+std::cos(angle)*y;
}
struct Sample { double distance=0; SdfStableId primitiveId=0; };

// Production scalar implementation of the normative SdfIr formulas. It has no
// renderer controls or dependency on the helper's independent test evaluator.
// A record is a pure function of its query point, so a record with several
// consumers answers each exact point once per query. A record with one consumer
// repeats only when that consumer does, and stays out of the memo.
class Evaluator final {
public:
  explicit Evaluator(const SdfIr& ir):root(ir.rootId) {
    for (const auto& r:ir.records) records.emplace(r.stableId,&r);
    for (const auto& outline:ir.textOutlines) outlines.emplace(outline.stableId,&outline);
    std::set<SdfStableId> consumed;
    for (const auto& r:ir.records) for (unsigned i=0;i<r.inputCount;++i)
      if (!consumed.insert(r.inputs[i]).second) shared.insert(r.inputs[i]);
  }
  Sample at(Point q) const { memo.clear(); return at(root,q); }
  // Records computed so far, not counting answers from the memo.
  std::uint64_t recordEvaluations() const { return evaluations; }
private:
  // Exact bits, so -0 and +0 stay apart: a formula can tell them apart.
  using MemoKey=std::pair<SdfStableId,std::array<std::uint64_t,3>>;
  Sample at(SdfStableId id,Point q) const {
    if(!shared.count(id)) return evaluate(id,q);
    MemoKey key{id,{}};
    std::memcpy(key.second.data(),q.data(),sizeof(q));
    const auto known=memo.find(key);
    if(known!=memo.end()) return known->second;
    const auto sample=evaluate(id,q);
    memo.emplace(key,sample);
    return sample;
  }
  Sample evaluate(SdfStableId id,Point q) const {
    ++evaluations;
    const auto& r=*records.at(id); const auto& p=r.parameters;
    const auto child=[&](Point v) { return at(r.inputs[0],v); };
    using O=SdfOperation;
    switch(r.operation) {
    case O::textExtrusion: {
      // The admitted immutable XY segments use nonzero winding, including
      // clockwise counters. Extrusion preserves Euclidean outside corners.
      const auto& outline=*outlines.at(r.resourceStableId);
      if(!textWork.charge(1,outline.segments.size())) return {std::numeric_limits<double>::quiet_NaN(),0};
      double squared=std::numeric_limits<double>::infinity(); int winding=0;
      for(const auto& segment:outline.segments) {
        const double dx=segment.x1-segment.x0,dy=segment.y1-segment.y0;
        const double px=q[0]-segment.x0,py=q[1]-segment.y0;
        const double t=std::clamp((px*dx+py*dy)/(dx*dx+dy*dy),0.0,1.0);
        const double ex=px-t*dx,ey=py-t*dy;
        squared=std::min(squared,ex*ex+ey*ey);
        const double side=dx*py-dy*px;
        if(segment.y0<=q[1] && segment.y1>q[1] && side>0) ++winding;
        if(segment.y0>q[1] && segment.y1<=q[1] && side<0) --winding;
      }
      const double xy=std::sqrt(squared)*(winding==0?1.0:-1.0),z=std::abs(q[2])-p[0]/2;
      return {std::hypot(std::max(xy,0.0),std::max(z,0.0))+std::min(std::max(xy,z),0.0),id}; }
    case O::sphere: return {length(q)-p[0],id};
    case O::box: return {box(q,{p[0],p[1],p[2]}),id};
    case O::roundedBox: return {box(q,{p[0]-p[3],p[1]-p[3],p[2]-p[3]})-p[3],id};
    case O::plane: { const Point n{p[0],p[1],p[2]}; return {dot(q,n)/length(n)+p[3],id}; }
    case O::torus: return {std::hypot(std::hypot(q[0],q[2])-p[0],q[1])-p[1],id};
    case O::capsule: {
      const Point a{p[0],p[1],p[2]},b{p[3],p[4],p[5]}; const auto ab=sub(b,a),v=sub(q,a);
      return {length(sub(v,mul(ab,std::clamp(dot(v,ab)/dot(ab,ab),0.0,1.0))))-p[6],id}; }
    case O::cylinder: {
      const auto a=std::hypot(q[0],q[2])-p[0],b=std::abs(q[1])-p[1];
      return {std::min(std::max(a,b),0.0)+std::hypot(std::max(a,0.0),std::max(b,0.0)),id}; }
    case O::cone: {
      const double x=std::hypot(q[0],q[2]),y=q[1],r0=p[0],h=p[1];
      const double ax=x-std::min(x,y<0?r0:0),ay=std::abs(y)-h;
      const double t=std::clamp(((r0-x)*r0+(-h-y)*(-2*h))/(r0*r0+4*h*h),0.0,1.0);
      const double bx=x-r0+r0*t,by=y+h-2*h*t;
      return {(bx<0 && ay<0?-1.0:1.0)*std::sqrt(std::min(ax*ax+ay*ay,bx*bx+by*by)),id}; }
    case O::gyroid:
      return {std::abs(std::sin(p[0]*q[0])*std::cos(p[0]*q[2])+std::sin(p[0]*q[1])*std::cos(p[0]*q[0])
          +std::sin(p[0]*q[2])*std::cos(p[0]*q[1]))/p[0]-p[1],id};
    case O::unionOp: case O::intersection: case O::subtraction:
    case O::smoothUnion: case O::smoothIntersection: case O::smoothSubtraction: {
      const auto a=at(r.inputs[0],q),b=at(r.inputs[1],q);
      const bool minimum=r.operation==O::unionOp || r.operation==O::smoothUnion;
      const bool cut=r.operation==O::subtraction || r.operation==O::smoothSubtraction;
      const double bv=cut?-b.distance:b.distance;
      const auto winner=(minimum?a.distance<=bv:a.distance>=bv)?a.primitiveId:b.primitiveId;
      double d=minimum?std::min(a.distance,bv):std::max(a.distance,bv);
      if(r.operation==O::smoothUnion) d=smooth(a.distance,b.distance,p[0]);
      if(r.operation==O::smoothIntersection) d=-smooth(-a.distance,-b.distance,p[0]);
      if(r.operation==O::smoothSubtraction) d=-smooth(-a.distance,b.distance,p[0]);
      return {d,winner}; }
    case O::translate: return child(sub(q,{p[0],p[1],p[2]}));
    case O::rotate: {
      Point n{p[0],p[1],p[2]}; n=mul(n,1/length(n)); const double c=std::cos(-p[3]),s=std::sin(-p[3]);
      return child(add(add(mul(q,c),mul(cross(n,q),s)),mul(n,dot(n,q)*(1-c)))); }
    case O::scale: {
      for(unsigned i=0;i<3;++i) q[i]/=p[i];
      auto a=child(q);
      a.distance*=std::min({std::abs(p[0]),std::abs(p[1]),std::abs(p[2])}); return a; }
    case O::repeat: for(unsigned i=0;i<3;++i) q[i]=modulo(q[i],p[i]); return child(q);
    case O::polarRepeat: {
      const unsigned axis=static_cast<unsigned>(p[0]),u=(axis+1)%3,v=(axis+2)%3;
      const auto radius=std::hypot(q[u],q[v]),angle=modulo(std::atan2(q[v],q[u])-p[2],6.2831853071795864769/p[1]);
      q[u]=radius*std::cos(angle); q[v]=radius*std::sin(angle); return child(q); }
    case O::mirror: for(unsigned i=0;i<3;++i) if(p[i]!=0) q[i]=std::abs(q[i]); return child(q);
    case O::twist: case O::bend: case O::taper: {
      const unsigned axis=static_cast<unsigned>(p[0]),u=(axis+1)%3,v=(axis+2)%3;
      const auto angle=-p[1]*q[axis];
      if(r.operation==O::twist) rotatePair(q[u],q[v],angle);
      if(r.operation==O::bend) rotatePair(q[u],q[axis],angle);
      if(r.operation==O::taper) { q[u]*=1+angle; q[v]*=1+angle; }
      return child(q); }
    case O::displacement: { auto a=child(q); a.distance+=p[0]*std::sin(p[1]*q[0])*std::sin(p[1]*q[1])*std::sin(p[1]*q[2]); return a; }
    case O::domainWarp: return child({q[0]+p[0]*std::sin(p[3]*q[1]),q[1]+p[1]*std::sin(p[4]*q[2]),q[2]+p[2]*std::sin(p[5]*q[0])});
    }
    return {std::numeric_limits<double>::quiet_NaN(),0};
  }
  SdfStableId root;
  std::map<SdfStableId,const SdfRecord*> records;
  std::map<SdfTextResourceId,const SdfTextOutline*> outlines;
  std::set<SdfStableId> shared;
  mutable std::map<MemoKey,Sample> memo;
  mutable std::uint64_t evaluations=0;
  mutable SdfTextWorkBudget textWork;
};
inline bool parametersValid(const SdfRecord& r) {
  using O=SdfOperation; const auto& p=r.parameters;
  const auto positive=[&](unsigned i,unsigned n) { for(;n;--n,++i) if(p[i]<0.000001) return false; return true; };
  const auto axis=[](double v) { return v==0 || v==1 || v==2; };
  switch(r.operation) {
  case O::textExtrusion: return p[0]>=kSdfTextMinimumDepth;
  case O::sphere: case O::smoothUnion: case O::smoothIntersection: case O::smoothSubtraction: return positive(0,1);
  case O::box: case O::repeat: return positive(0,3);
  case O::roundedBox: return positive(0,4) && p[3]<=std::min({p[0],p[1],p[2]});
  case O::plane: case O::rotate: return length({p[0],p[1],p[2]})>=0.000001;
  case O::torus: case O::cylinder: case O::cone: case O::gyroid: return positive(0,2);
  case O::capsule: return positive(6,1) && length({p[3]-p[0],p[4]-p[1],p[5]-p[2]})>=0.000001;
  case O::scale: return std::abs(p[0])>=0.000001 && std::abs(p[1])>=0.000001 && std::abs(p[2])>=0.000001;
  case O::polarRepeat: return axis(p[0]) && p[1]>=1 && p[1]<=1024 && std::floor(p[1])==p[1];
  case O::mirror: return (p[0]==0 || p[0]==1) && (p[1]==0 || p[1]==1) && (p[2]==0 || p[2]==1) && p[0]+p[1]+p[2]>0;
  case O::twist: case O::bend: case O::taper: return axis(p[0]);
  case O::displacement: return positive(1,1);
  case O::domainWarp: return positive(3,3);
  default: return true;
  }
}
} // namespace detail
struct Budget final {
  std::size_t samples=0, cells=0, edgeCapacity=0, triangleCapacity=0, recordWork=0, sampleWork=0, temporaryBytes=0;
  std::uint64_t textRecordVisits=0, textSegmentVisits=0;
};
inline bool inspect(const Request& r,Budget& budget,std::string& error) {
  const auto fail=[&](const char* e) { error=e; return false; };
  if(r.algorithmVersion!=kAlgorithmVersion
      || (r.geometry.schemaVersion!=kSdfIrSchemaVersion && r.geometry.schemaVersion!=kSdfIrTextSchemaVersion)
      || r.geometry.records.empty() || r.geometry.records.size()>kMaximumRecords
      || r.geometry.rootId==0 || r.geometry.rootId>kMaximumStableId) return fail("SDF surface has unsupported algorithm or IR identity");
  if(r.geometry.schemaVersion==kSdfIrTextSchemaVersion) {
    if(!validateSdfTextIrResources(r.geometry,error)) return false;
  } else {
    if(!r.geometry.textOutlines.empty()) return fail("Legacy SDF surface IR cannot own text outlines");
    for(const auto& record:r.geometry.records) if(record.resourceStableId!=0)
      return fail("Legacy SDF surface IR cannot reference text outlines");
  }
  budget={}; budget.samples=1; budget.cells=1;
  double spacing=std::numeric_limits<double>::max();
  for(unsigned i=0;i<3;++i) {
    if(!std::isfinite(r.minimum[i]) || !std::isfinite(r.maximum[i]) || std::abs(r.minimum[i])>1000000
        || std::abs(r.maximum[i])>1000000 || r.minimum[i]>=r.maximum[i] || r.cells[i]<2)
      return fail("SDF surface requires a finite AABB and at least 2 cells on each axis");
    spacing=std::min(spacing,(r.maximum[i]-r.minimum[i])/r.cells[i]);
    const std::size_t axisSamples=std::size_t{r.cells[i]}+1;
    if(axisSamples>kMaximumSamples/budget.samples) return fail("SDF surface grid exceeds its 32-bit sample index");
    budget.samples*=axisSamples; budget.cells*=r.cells[i];
  }
  if(!std::isfinite(r.gradientStep) || r.gradientStep<=0 || r.gradientStep>spacing/4
      || !std::isfinite(r.maximumResidual) || r.maximumResidual<=0 || r.maximumResidual>spacing)
    return fail("SDF surface gradient step or residual limit disagrees with grid spacing");
  std::map<SdfStableId,const SdfRecord*> records;
  for(const auto& v:r.geometry.records) {
    const auto shape=sdfOperationSchema(v.operation,r.geometry.schemaVersion);
    if(v.stableId==0 || v.stableId>kMaximumStableId || !records.emplace(v.stableId,&v).second
        || !shape.valid || v.inputCount!=shape.inputCount || v.parameterCount!=shape.parameterCount)
      return fail("SDF surface IR has an invalid record identity or shape");
    for(unsigned i=0;i<v.inputs.size();++i) if(i>=v.inputCount && v.inputs[i]!=0)
      return fail("SDF surface IR has unused input data");
    for(unsigned i=0;i<v.parameters.size();++i)
      if(!std::isfinite(v.parameters[i]) || std::abs(v.parameters[i])>1000000 || (i>=v.parameterCount && v.parameters[i]!=0))
        return fail("SDF surface IR has invalid parameter data");
    if(!detail::parametersValid(v)) return fail("SDF surface IR parameters violate the normative operation");
  }
  std::set<SdfStableId> active,visited;
  std::map<SdfStableId,std::pair<std::size_t,std::size_t>> costs;
  std::function<bool(SdfStableId)> visit=[&](SdfStableId id) {
    if(costs.count(id)) return true;
    if(!records.count(id) || !active.insert(id).second) return false;
    std::size_t work=1,depth=1;
    const auto& v=*records.at(id);
    for(unsigned i=0;i<v.inputCount;++i) {
      if(!visit(v.inputs[i])) return false;
      const auto child=costs.at(v.inputs[i]);
      work=sdfSaturatingAdd(work,child.first); depth=std::max(depth,child.second+1);
    }
    if(depth>64) return false;
    active.erase(id); visited.insert(id); costs[id]={work,depth}; return true;
  };
  if(!visit(r.geometry.rootId) || visited.size()!=records.size()) return fail("SDF surface IR is cyclic, disconnected or exceeds the depth limit");
  budget.recordWork=costs.at(r.geometry.rootId).first;
  // Seven directed lattice edges cover axis, face-diagonal and body-diagonal
  // edges of the fixed six-tetrahedron Freudenthal subdivision.
  budget.edgeCapacity=7*budget.samples; budget.triangleCapacity=12*budget.cells;
  const auto evaluations=budget.samples+2*budget.cells+7*budget.edgeCapacity+budget.triangleCapacity;
  budget.sampleWork=sdfSaturatingMultiply(evaluations,budget.recordWork);
  if(r.geometry.schemaVersion==kSdfIrTextSchemaVersion) {
    SdfTextWorkBudget textWork;
    if(!chargeSdfTextIrEvaluationWork(r.geometry,evaluations,textWork,error)) return false;
    budget.textRecordVisits=textWork.recordVisits; budget.textSegmentVisits=textWork.segmentVisits;
  }
  // Includes map nodes, temporary samples and conservative output arrays.
  budget.temporaryBytes=budget.samples*sizeof(double)+budget.edgeCapacity*256+budget.triangleCapacity*64
      +r.geometry.records.size()*sizeof(SdfRecord);
  // Retained outline objects, vector padding, strings, lookup nodes and the
  // largest temporary identity buffer used during validation. Evaluation only
  // references these immutable resources; it does not copy or reshape them.
  std::size_t identityBytes=0;
  for(const auto& outline:r.geometry.textOutlines) {
    budget.temporaryBytes+=sizeof(SdfTextOutline)+256+outline.sourceText.size()+outline.sourceTextSha256.size()
        +outline.embeddedFontToken.size()+outline.embeddedFontSha256.size()
        +outline.glyphs.size()*sizeof(SdfTextGlyph)+outline.contours.size()*sizeof(SdfTextContour)
        +outline.segments.size()*sizeof(SdfTextSegment);
    identityBytes=std::max(identityBytes,sdfTextOutlinePayloadBytes(outline));
  }
  budget.temporaryBytes+=identityBytes*2;
  error.clear(); return true;
}
struct Mesh final {
  std::vector<std::array<float,3>> positions,normals,uv;
  std::vector<std::uint64_t> vertexIds,primitiveIds;
  std::vector<std::uint32_t> indices;
};

// Algorithm 1 clips the zero set to the authored box without adding caps.
// Cells run Z/Y/X; each cube uses the six tetrahedra below, sharing diagonals
// across adjacent cubes. Negative is inside. An isolated exact zero uses its
// grid-point identity; zero-valued edges refuse as ambiguous. Crossing edges
// use linear interpolation. A 2/2 tetra splits its quad by the stated order.
// Normals use central differences at gradientStep, UV is normalized box X/Z.
// The residual limit checks retained float crossings and triangle centers.
// Face provenance evaluates the full winning primitive at the triangle center,
// with first-input CSG ties. This sampled mesh does not certify exact topology.
inline bool materialize(const Request& r,std::uint64_t producer,std::size_t maxVertices,
    std::size_t maxIndices,Mesh& out,std::string& error) try {
  Budget budget;
  if(producer==0 || producer>kMaximumStableId) { error="SDF surface producer identity is invalid"; return false; }
  if(!inspect(r,budget,error)) return false;
  const auto fail=[&](const char* e) { error=e; return false; };
  detail::Evaluator evaluator(r.geometry);
  const std::uint32_t nx=r.cells[0]+1,ny=r.cells[1]+1;
  const auto id=[&](unsigned x,unsigned y,unsigned z) { return (z*ny+y)*nx+x; };
  const auto point=[&](std::uint32_t i) {
    const unsigned x=i%nx,y=(i/nx)%ny,z=i/(nx*ny);
    return detail::Point{r.minimum[0]+(r.maximum[0]-r.minimum[0])*x/r.cells[0],
      r.minimum[1]+(r.maximum[1]-r.minimum[1])*y/r.cells[1],r.minimum[2]+(r.maximum[2]-r.minimum[2])*z/r.cells[2]};
  };
  std::vector<double> samples(budget.samples);
  for(std::uint32_t i=0;i<samples.size();++i) {
    samples[i]=evaluator.at(point(i)).distance;
    if(!std::isfinite(samples[i])) return fail("SDF surface evaluator returned a non-finite sample");
  }
  using Edge=std::pair<std::uint32_t,std::uint32_t>;
  std::map<Edge,std::uint32_t> edges;
  std::size_t triangleCount=0;
  constexpr unsigned tetrahedra[6][4]={{0,1,3,7},{0,3,2,7},{0,2,6,7},{0,6,4,7},{0,4,5,7},{0,5,1,7}};
  const auto edge=[&](std::uint32_t a,std::uint32_t b) {
    if(samples[a]==0) return Edge{a,a};
    if(samples[b]==0) return Edge{b,b};
    return Edge{std::min(a,b),std::max(a,b)};
  };
  // Count and intern canonical crossings first. No output allocation occurs
  // until exact vertex/index cardinalities have passed their capacity checks.
  const auto traverse=[&](const auto& triangle) {
    for(unsigned z=0;z<r.cells[2];++z) for(unsigned y=0;y<r.cells[1];++y) for(unsigned x=0;x<r.cells[0];++x) {
      std::array<std::uint32_t,8> cube{};
      for(unsigned i=0;i<8;++i) cube[i]=id(x+(i&1),y+((i>>1)&1),z+((i>>2)&1));
      bool same=true; for(unsigned i=1;i<8;++i) same &= (samples[cube[0]]<0)==(samples[cube[i]]<0);
      if(same) {
        const auto center=detail::mul(detail::add(point(cube[0]),point(cube[7])),0.5);
        const auto d=evaluator.at(center).distance;
        if(!std::isfinite(d) || ((d<0)!=(samples[cube[0]]<0))) return false;
      }
      for(const auto& t:tetrahedra) {
        std::array<std::uint32_t,4> inside{},outside{}; unsigned ni=0,no=0;
        for(unsigned i=0;i<4;++i) {
          const auto a=cube[t[i]];
          for(unsigned j=0;j<i;++j) if(samples[a]==0 && samples[cube[t[j]]]==0) return false;
          if(samples[a]<0) inside[ni++]=a; else outside[no++]=a;
        }
        if(ni==0 || no==0) continue;
        const auto emit=[&](Edge a,Edge b,Edge c) {
          if(a==b || b==c || a==c) return true;
          return triangle(std::array<Edge,3>{a,b,c});
        };
        if(ni==1) { if(!emit(edge(inside[0],outside[0]),edge(inside[0],outside[1]),edge(inside[0],outside[2]))) return false; }
        else if(no==1) { if(!emit(edge(outside[0],inside[0]),edge(outside[0],inside[1]),edge(outside[0],inside[2]))) return false; }
        else {
          const auto a=edge(inside[0],outside[0]),b=edge(inside[0],outside[1]),c=edge(inside[1],outside[0]),d=edge(inside[1],outside[1]);
          if(!emit(a,b,d) || !emit(a,d,c)) return false;
        }
      }
    }
    return true;
  };
  if(!traverse([&](const std::array<Edge,3>& t) {
      if(triangleCount>=maxIndices/3) return false;
      ++triangleCount;
      for(const auto& e:t) {
        if(!edges.count(e)) {
          if(edges.size()>=maxVertices) return false;
          edges.emplace(e,static_cast<std::uint32_t>(edges.size()));
        }
      }
      return true;
    })) return fail("SDF surface is ambiguous, under-resolved or exceeds vertex/index capacity");
  if(edges.empty() || triangleCount==0) return fail("SDF surface has no resolved zero set in its domain");
  Mesh result;
  result.positions.resize(edges.size()); result.vertexIds.resize(edges.size());
  result.normals.resize(edges.size()); result.uv.resize(edges.size());
  result.indices.reserve(triangleCount*3); result.primitiveIds.reserve(triangleCount);
  std::set<std::uint64_t> stableIds;
  for(const auto& entry:edges) {
    const auto [a,b]=entry.first; const auto index=entry.second;
    const auto p0=point(a),p1=point(b);
    const double weight=a==b?0:samples[a]/(samples[a]-samples[b]);
    const auto p=detail::add(p0,detail::mul(detail::sub(p1,p0),weight));
    for(unsigned i=0;i<3;++i) result.positions[index][i]=static_cast<float>(p[i]);
    const detail::Point retained{result.positions[index][0],result.positions[index][1],result.positions[index][2]};
    const auto residual=evaluator.at(retained).distance;
    if(!std::isfinite(residual) || std::abs(residual)>r.maximumResidual) return fail("SDF surface edge residual requires a finer grid");
    detail::Point normal{};
    for(unsigned i=0;i<3;++i) {
      auto plus=retained,minus=retained; plus[i]+=r.gradientStep; minus[i]-=r.gradientStep;
      normal[i]=evaluator.at(plus).distance-evaluator.at(minus).distance;
    }
    const auto n=detail::length(normal);
    if(!std::isfinite(n) || n<=1e-15) return fail("SDF surface gradient is undefined at a crossing");
    for(unsigned i=0;i<3;++i) result.normals[index][i]=static_cast<float>(normal[i]/n);
    result.uv[index]={static_cast<float>((retained[0]-r.minimum[0])/(r.maximum[0]-r.minimum[0])),
      static_cast<float>((retained[2]-r.minimum[2])/(r.maximum[2]-r.minimum[2])),0};
    std::uint64_t hash=1469598103934665603ull;
    for(const std::uint64_t value:{producer,static_cast<std::uint64_t>(r.algorithmVersion),
        static_cast<std::uint64_t>(r.cells[0]),static_cast<std::uint64_t>(r.cells[1]),static_cast<std::uint64_t>(r.cells[2]),
        static_cast<std::uint64_t>(a),static_cast<std::uint64_t>(b)})
      for(unsigned shift=0;shift<64;shift+=8) { hash^=(value>>shift)&255; hash*=1099511628211ull; }
    hash&=kMaximumStableId;
    if(hash==0 || !stableIds.insert(hash).second) return fail("SDF surface canonical edge identity collision");
    result.vertexIds[index]=hash;
  }
  if(!traverse([&](const std::array<Edge,3>& t) {
      std::array<std::uint32_t,3> indices{edges.at(t[0]),edges.at(t[1]),edges.at(t[2])};
      const auto p=[&](unsigned i) { const auto& v=result.positions[indices[i]]; return detail::Point{v[0],v[1],v[2]}; };
      const auto a=p(0),b=p(1),c=p(2),cross=detail::cross(detail::sub(b,a),detail::sub(c,a));
      detail::Point normal{};
      for(const auto i:indices) for(unsigned j=0;j<3;++j) normal[j]+=result.normals[i][j];
      const auto orientation=detail::dot(cross,normal);
      if(!std::isfinite(orientation) || detail::length(cross)<=1e-14 || std::abs(orientation)<=1e-16) return false;
      if(orientation<0) std::swap(indices[1],indices[2]);
      const auto sample=evaluator.at(detail::mul(detail::add(detail::add(a,b),c),1.0/3));
      if(!std::isfinite(sample.distance) || std::abs(sample.distance)>r.maximumResidual || sample.primitiveId==0) return false;
      result.indices.insert(result.indices.end(),indices.begin(),indices.end()); result.primitiveIds.push_back(sample.primitiveId);
      return true;
    })) return fail("SDF surface contains a degenerate, ambiguous or under-resolved triangle");
  out=std::move(result); error.clear(); return true;
} catch(const std::bad_alloc&) {
  error="SDF surface could not allocate its samples and mesh"; return false;
} catch(const std::length_error&) {
  error="SDF surface could not allocate its samples and mesh"; return false;
}
} // namespace videowire::sdfsurface
