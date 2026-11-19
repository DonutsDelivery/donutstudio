#pragma once

#include <array>
#include <string_view>

namespace visualsimulation
{
// Existing graph parameter names remain the destination identity. Entries are
// append-only: their indices are persisted by the existing VideoParam mapping.
struct Parameter { const char* name; float minimum, maximum, fallback; };
inline constexpr std::array<Parameter, 22> parameters {{
    {"gravity",-4,4,0}, {"gravityX",-4,4,0}, {"gravityZ",-4,4,0},
    {"drag",0,10,0}, {"angularDrag",0,10,0}, {"restitution",0,1,.7f},
    {"impulseX",-4,4,0}, {"impulseY",-4,4,0}, {"impulseZ",-4,4,0},
    {"attraction",0,10,0}, {"linkSpring",0,10,0}, {"linkRatioInfluence",0,1,1},
    {"linkRestScale",.1f,4,1}, {"bodyMass",.01f,100,1}, {"friction",0,1,0},
    {"bodyRadius",.001f,.1f,.015f}, {"speed",0,4,1}, {"rmsGain",0,8,0},
    {"onsetGain",0,8,0}, {"emissionRate",-1,64,-1}, {"reset",0,1,0},
    {"resetTime",0,86400,0}
}};
inline bool isParameter(std::string_view name)
{
    for (const auto& parameter : parameters) if (name == parameter.name) return true;
    return false;
}
}
