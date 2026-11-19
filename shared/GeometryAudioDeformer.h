#pragma once

#include "GeometryCore.h"

namespace videowire::geometry {
// Positions change in place; indices and stable vertex identities remain intact.
// responses holds one pre-falloff amount per vertex. History rows supply
// different amounts per vertex, while a current frame supplies one shared value.
inline bool applyAudioDeformerResponses(GeometryData& mesh, const spectrum::Binding& binding,
                                        const std::vector<double>& responses, double seconds,
                                        std::string& error) {
  if (binding.target != spectrum::Target::geometryDeformation
      || !spectrum::validate(binding, mesh.positions.size(), error)) return false;
  if (responses.size() != mesh.positions.size()) {
    error = "Audio Deformer response count disagrees with its vertices"; return false;
  }
  const auto& d = binding.deformer;
  std::vector<Vec3> normals(mesh.positions.size());
  for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
    const auto a = mesh.indices[i], b = mesh.indices[i+1], c = mesh.indices[i+2];
    if (a >= normals.size() || b >= normals.size() || c >= normals.size()) {
      error = "Audio Deformer received an invalid triangle index"; return false;
    }
    const auto& p = mesh.positions[a]; const auto& q = mesh.positions[b]; const auto& r = mesh.positions[c];
    const Vec3 u{q.x-p.x,q.y-p.y,q.z-p.z}, v{r.x-p.x,r.y-p.y,r.z-p.z};
    const Vec3 n{u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x};
    for (const auto vertex : {a,b,c}) {
      normals[vertex].x += n.x; normals[vertex].y += n.y; normals[vertex].z += n.z;
    }
  }
  const double phase = d.phase + seconds * d.speed;
  for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
    auto& p = mesh.positions[i];
    const double x = p.x-d.center[0], y = p.y-d.center[1], z = p.z-d.center[2];
    const double radius = std::sqrt(x*x+y*y+z*z);
    const double falloff = d.falloffRadius > 0
        ? std::pow(std::clamp(1.0-radius/d.falloffRadius,0.0,1.0),d.falloffPower) : 1.0;
    const double amount = responses[i] * falloff;
    auto normal = normals[i];
    const double length = std::sqrt(static_cast<double>(normal.x)*normal.x
        + static_cast<double>(normal.y)*normal.y + static_cast<double>(normal.z)*normal.z);
    normal = length > 0 ? Vec3{static_cast<float>(normal.x/length),static_cast<float>(normal.y/length),
                              static_cast<float>(normal.z/length)} : Vec3{0,0,1};
    double dx=0,dy=0,dz=0;
    const auto alongNormal = [&](double distance) {
      dx=normal.x*distance; dy=normal.y*distance; dz=normal.z*distance;
    };
    switch (d.mode) {
      case spectrum::Deformation::normal: alongNormal(amount); break;
      case spectrum::Deformation::vector:
        dx=amount*binding.direction[0]; dy=amount*binding.direction[1]; dz=amount*binding.direction[2]; break;
      case spectrum::Deformation::bulge: alongNormal(amount*std::exp(-radius*radius)); break;
      case spectrum::Deformation::ripple: alongNormal(amount*std::sin(radius*d.frequency-phase)); break;
      case spectrum::Deformation::wave: alongNormal(amount*std::sin(y*d.frequency+phase)); break;
      case spectrum::Deformation::twist: {
        const double angle=amount*y;
        dx=x*std::cos(angle)+z*std::sin(angle)-x;
        dz=-x*std::sin(angle)+z*std::cos(angle)-z; break;
      }
      case spectrum::Deformation::bend:
        if (std::abs(amount)>1.0e-6) {
          const double angle=amount*y, inverse=1.0/amount;
          dx=(x+inverse)*std::cos(angle)-inverse-x;
          dy=(x+inverse)*std::sin(angle)-y;
        }
        break;
      case spectrum::Deformation::taper: {
        const double scale=std::clamp(1.0+amount*y,0.01,100.0);
        dx=x*(scale-1); dz=z*(scale-1); break;
      }
      case spectrum::Deformation::squash: {
        const double strain=std::clamp(amount,-4.0,4.0);
        dx=x*(std::exp(strain*0.5)-1); dy=y*(std::exp(-strain)-1); dz=z*(std::exp(strain*0.5)-1); break;
      }
      case spectrum::Deformation::turbulence:
        dx=amount*std::sin(y*d.frequency+std::sin(z*d.frequency+phase));
        dy=amount*std::sin(z*d.frequency+std::sin(x*d.frequency+phase+2.1));
        dz=amount*std::sin(x*d.frequency+std::sin(y*d.frequency+phase+4.2)); break;
      case spectrum::Deformation::attraction: {
        const double strength=std::clamp(amount,-1.0,1.0);
        dx=-x*strength; dy=-y*strength; dz=-z*strength; break;
      }
    }
    p={static_cast<float>(p.x+dx),static_cast<float>(p.y+dy),static_cast<float>(p.z+dz)};
    if (!finite(p)) { error="Audio Deformer produced a non-finite position"; return false; }
  }
  error.clear(); return true;
}

inline bool applyAudioDeformer(GeometryData& mesh, const spectrum::Binding& binding,
                               const spectrum::Bands& bands, double seconds, std::string& error) {
  if (binding.target != spectrum::Target::geometryDeformation
      || !spectrum::validate(binding, mesh.positions.size(), error)) return false;
  const std::vector<double> responses(mesh.positions.size(), spectrum::sample(binding, bands, 0));
  return applyAudioDeformerResponses(mesh, binding, responses, seconds, error);
}
} // namespace videowire::geometry
