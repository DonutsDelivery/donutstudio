// video_param_grammar.h — the single source of truth for the render-graph param
// grammar (Stage 6: "collapse the three video-param-grammar sites into one resolver").
//
// A mod-matrix / automation destination is "clip<id>/<node>/<param>"; this maps a
// (node, param) pair onto a field of a per-clip param struct. Before Stage 6 that map
// was copy-pasted in THREE places — exporter.cpp applyGraphParam / getGraphParam and
// viewport.cpp applyTo / readFrom — guarded only by "KEEP IN SYNC" comments, a silent-
// divergence hazard (a routing that worked in the live viewport could write a different
// field, or nothing, on export). This header is now the one definition both callers use.
//
// It is templated over the param struct P because the exporter (ClipRenderState) and
// the viewport (ClipGraphParams) deliberately use distinct — but field-identical —
// structs. The template makes the compiler enforce that they stay field-compatible:
// rename or drop a field on one and the other's instantiation fails to build. It covers
// the transform2d / source / mask / effect<N> namespaces common to both callers; the
// exporter-only "gen" (ISF input) namespace stays caller-side glue (it touches a
// std::map the viewport struct does not have).
//
// Dependency-light on purpose: needs only effect_defs.h (no GL, no FFmpeg), so the
// grammar is unit-testable standalone — see tests/param_grammar_tests.cpp.

#pragma once

#include "effect_defs.h"

#include <algorithm>
#include <cstring>
#include <string>

namespace videoparam
{

// Zero an effect slot's params, then seed the type's table defaults. Duck-typed over
// any slot exposing `.type` (int) and `.params` (float array), so it needs no GL header
// (the concrete slot type, videorender::EffectSlotState, lives behind one).
template <class Slot>
inline void resetSlotParams (Slot& e)
{
    std::memset (e.params, 0, sizeof (e.params));
    if (const auto* def = videofx::effectDefFor (e.type))
        for (int i = 0; i < def->paramCount; ++i)
            e.params[i] = def->params[i].defaultValue;
}

// Write one render-graph param onto P. Returns false for an unknown node/param so the
// caller can try its own namespaces (e.g. the exporter's "gen"). Same clamping the
// three original sites used.
template <class P>
inline bool applyGraphParam (P& p, const std::string& node,
                             const std::string& param, double value)
{
    if (node == "transform2d")
    {
        if (param == "scale")      { p.scale = (float) value; return true; }
        if (param == "translateX") { p.translateX = (float) value; return true; }
        if (param == "translateY") { p.translateY = (float) value; return true; }
        if (param == "rotation")   { p.rotationDeg = (float) value; return true; }
        if (param == "cropLeft")   { p.cropLeft = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "cropRight")  { p.cropRight = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "cropTop")    { p.cropTop = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "cropBottom") { p.cropBottom = (float) std::clamp (value, 0.0, 1.0); return true; }
    }
    else if (node == "source")
    {
        if (param == "opacity")   { p.opacity = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "visible")   { p.visible = value >= 0.5 ? 1.0f : 0.0f; return true; }
        if (param == "zOrder")    { p.zOrder = (float) value; return true; }
        if (param == "blendMode") { p.blendMode = (int) value; return true; }
    }
    else if (node == "mask")
    {
        if (param == "type")    { p.maskType = std::clamp ((int) value, 0, 2); return true; }
        if (param == "cx")      { p.maskCx = (float) value; return true; }
        if (param == "cy")      { p.maskCy = (float) value; return true; }
        if (param == "w")       { p.maskW = (float) std::max (value, 0.0); return true; }
        if (param == "h")       { p.maskH = (float) std::max (value, 0.0); return true; }
        if (param == "feather") { p.maskFeather = (float) std::clamp (value, 0.0, 1.0); return true; }
        if (param == "invert")  { p.maskInvert = value >= 0.5 ? 1 : 0; return true; }
    }
    else if (node.rfind ("effect", 0) == 0 && node.size() > 6)
    {
        int slot = -1;
        try { slot = std::stoi (node.substr (6)); }
        catch (...) { return false; }
        if (slot < 0 || slot >= videofx::kMaxEffectSlots) return false;

        auto& e = p.effects[slot];
        if (param == "type")
        {
            const int t = (int) value;
            e.type = (t >= 0 && t < videofx::kEffectTypeCount) ? t : -1;
            resetSlotParams (e);   // table defaults for the new type
            return true;
        }
        if (param == "enabled") { e.enabled = value >= 0.5; return true; }
        if (e.type < 0) return false;
        const int idx = videofx::effectParamIndex (e.type, param.c_str());
        if (idx < 0) return false;
        const auto& pd = videofx::kEffectDefs[e.type].params[idx];
        e.params[idx] = std::clamp ((float) value, pd.minValue, pd.maxValue);
        return true;
    }
    return false;
}

// Read one render-graph param from P — the inverse of applyGraphParam, used to fetch
// the `base` a mod routing modulates (combine(mode, base, value)). Returns false for
// any param applyGraphParam does not write, so a routing to an unknown/unreadable param
// is skipped rather than laid over a junk base.
template <class P>
inline bool getGraphParam (const P& p, const std::string& node,
                           const std::string& param, double& out)
{
    if (node == "transform2d")
    {
        if (param == "scale")      { out = p.scale;       return true; }
        if (param == "translateX") { out = p.translateX;  return true; }
        if (param == "translateY") { out = p.translateY;  return true; }
        if (param == "rotation")   { out = p.rotationDeg; return true; }
        if (param == "cropLeft")   { out = p.cropLeft;    return true; }
        if (param == "cropRight")  { out = p.cropRight;   return true; }
        if (param == "cropTop")    { out = p.cropTop;     return true; }
        if (param == "cropBottom") { out = p.cropBottom;  return true; }
    }
    else if (node == "source")
    {
        if (param == "opacity")   { out = p.opacity;            return true; }
        if (param == "visible")   { out = p.visible;            return true; }
        if (param == "zOrder")    { out = p.zOrder;             return true; }
        if (param == "blendMode") { out = (double) p.blendMode; return true; }
    }
    else if (node == "mask")
    {
        if (param == "type")    { out = (double) p.maskType;   return true; }
        if (param == "cx")      { out = p.maskCx;              return true; }
        if (param == "cy")      { out = p.maskCy;              return true; }
        if (param == "w")       { out = p.maskW;               return true; }
        if (param == "h")       { out = p.maskH;               return true; }
        if (param == "feather") { out = p.maskFeather;         return true; }
        if (param == "invert")  { out = (double) p.maskInvert; return true; }
    }
    else if (node.rfind ("effect", 0) == 0 && node.size() > 6)
    {
        int slot = -1;
        try { slot = std::stoi (node.substr (6)); }
        catch (...) { return false; }
        if (slot < 0 || slot >= videofx::kMaxEffectSlots) return false;
        const auto& e = p.effects[slot];
        if (param == "type")    { out = (double) e.type;         return true; }
        if (param == "enabled") { out = e.enabled ? 1.0 : 0.0;   return true; }
        if (e.type < 0) return false;
        const int idx = videofx::effectParamIndex (e.type, param.c_str());
        if (idx < 0) return false;
        out = e.params[idx];
        return true;
    }
    return false;
}

} // namespace videoparam
