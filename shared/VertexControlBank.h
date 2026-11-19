#pragma once

#include "VertexModifierIr.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

// The vertex control bank: one float per Vertex Control Parameter slot, read by
// every engine (OpenGL and Metal from a per-draw texture, Path Trace and the
// Geometry Core vertex stage on the host). Each record carries its slot and
// authored value; a published control or its modulation arrives per frame as
// runtime parameter "visual<record stable id>/value" and replaces the authored
// value without touching the program, so the shader text never changes.
namespace videowire::vertexcontrol
{
inline constexpr const char* kValueParameter = "value";
inline constexpr double kMaximumAbsoluteValue = 1000000.0;

inline std::string runtimeKey (VertexModifierStableId stableId)
{
    return "visual" + std::to_string (stableId) + "/" + kValueParameter;
}

// Fills `bank` with max slot + 1 values, or leaves it empty for a program with
// no control reads. Two records on one slot must agree, or the bank is refused
// by name rather than picking one.
inline bool bank (const VertexModifierIr& ir, const std::map<std::string, double>* runtime,
                  std::vector<float>& bank, std::string& error)
{
    bank.clear();
    std::vector<VertexModifierStableId> owner;
    for (const auto& record : ir.records)
    {
        if (record.operation != VertexModifierOperation::controlParameter) continue;
        const auto index = record.parameters[0];
        if (! (index >= 0 && index < 4096 && std::floor (index) == index))
        {
            error = "Vertex Control Parameter slot is not an index from 0 to 4095";
            return false;
        }
        double value = record.parameterCount > 1 ? record.parameters[1] : 0.0;
        if (runtime != nullptr)
            if (const auto found = runtime->find (runtimeKey (record.stableId)); found != runtime->end())
                value = found->second;
        if (! std::isfinite (value))
        {
            error = "Vertex Control Parameter value is not finite";
            return false;
        }
        const auto slot = static_cast<std::size_t> (index);
        const auto stored = static_cast<float> (std::clamp (value, -kMaximumAbsoluteValue, kMaximumAbsoluteValue));
        if (slot >= bank.size()) { bank.resize (slot + 1, 0.0f); owner.resize (slot + 1, 0); }
        if (owner[slot] != 0 && bank[slot] != stored)
        {
            error = "Two Vertex Control Parameter nodes read slot " + std::to_string (slot) + " with different values";
            return false;
        }
        bank[slot] = stored;
        owner[slot] = record.stableId;
    }
    error.clear();
    return true;
}

// The RGBA32F texels OpenGL and Metal read: slot i sits in texel i / 4, lane i % 4.
inline std::vector<float> texels (const std::vector<float>& bank)
{
    std::vector<float> result ((bank.size() + 3) / 4 * 4, 0.0f);
    std::copy (bank.begin(), bank.end(), result.begin());
    return result;
}

// Two rows for the draw: row 0 is this frame's bank and row 1 the prior frame's,
// which the motion pass evaluates the previous position with. A prior bank of
// another size belongs to another program, so the row repeats this frame's.
inline std::vector<float> motionTexels (const std::vector<float>& bank, const std::vector<float>* prior)
{
    auto result = texels (bank);
    const auto previous = prior != nullptr && prior->size() == bank.size() ? texels (*prior) : result;
    result.insert (result.end(), previous.begin(), previous.end());
    return result;
}
} // namespace videowire::vertexcontrol
