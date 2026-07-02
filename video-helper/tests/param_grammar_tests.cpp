// param_grammar_tests.cpp — unit tests for the shared render-graph param grammar
// (video_param_grammar.h, the Stage-6 collapse of the three duplicated grammar sites).
//
// Zero test-framework dependency (a tiny CHECK harness, like the mod-engine tests).
// Returns the failure count as the process exit code. The grammar header is templated
// over the param struct, so this test drives it through a MOCK struct that mirrors the
// real ClipRenderState / ClipGraphParams fields — the real structs are field-checked at
// compile time when the helper builds (a renamed/dropped field fails to instantiate).
//
// What it pins:
//   - round-trip: apply(node,param,v) then get(node,param) returns v (every namespace)
//   - clamping: opacity/crop/feather clamp to [0,1]; mask w/h clamp to >=0
//   - bool grammar: visible/invert use the >= 0.5 threshold
//   - effect slot: setting "type" seeds the table defaults; a param clamps to the
//     effect's declared min/max; an out-of-range slot is rejected
//   - unknown node/param returns false (so a caller's extra namespace, e.g. "gen",
//     still gets its turn)

#include "../src/video_param_grammar.h"

#include <cmath>
#include <cstdio>
#include <string>

static int g_fail = 0;
static int g_total = 0;

static void check (bool cond, const std::string& what)
{
    ++g_total;
    if (! cond) { ++g_fail; std::printf ("  FAIL: %s\n", what.c_str()); }
}

static void checkNear (double a, double b, double eps, const std::string& what)
{
    ++g_total;
    if (std::fabs (a - b) > eps)
    {
        ++g_fail;
        std::printf ("  FAIL: %s  (got %.6f, expected %.6f +/- %.6f)\n", what.c_str(), a, b, eps);
    }
}

// A minimal stand-in for ClipRenderState / ClipGraphParams (same field names + an
// effects[] of slots exposing .type/.enabled/.params, which is all the grammar touches).
struct MockSlot
{
    int   type    = -1;
    bool  enabled = false;
    float params[videofx::kMaxEffectParams] = {};
};

struct MockClip
{
    float scale = 1.0f;
    float translateX = 0.0f, translateY = 0.0f;
    float rotationDeg = 0.0f;
    float cropLeft = 0.0f, cropRight = 0.0f, cropTop = 0.0f, cropBottom = 0.0f;
    float opacity = 1.0f;
    float visible = 1.0f;
    float zOrder = 0.0f;
    int   blendMode = 0;
    MockSlot effects[videofx::kMaxEffectSlots];
    int   maskType = 0;
    float maskCx = 0.5f, maskCy = 0.5f;
    float maskW = 0.8f, maskH = 0.8f;
    float maskFeather = 0.05f;
    int   maskInvert = 0;
};

// apply then read-back; assert the read equals the expected stored value.
static void roundTrip (const char* node, const char* param, double in, double expected)
{
    MockClip c;
    const bool wrote = videoparam::applyGraphParam (c, node, param, in);
    check (wrote, std::string ("apply ") + node + "/" + param);
    double out = -1234.0;
    const bool read = videoparam::getGraphParam (c, node, param, out);
    check (read, std::string ("get ") + node + "/" + param);
    checkNear (out, expected, 1e-5, std::string ("round-trip ") + node + "/" + param);
}

int main()
{
    std::printf ("video param grammar tests\n");

    // transform2d — pass-through (no clamp) and crop (clamped to [0,1]).
    roundTrip ("transform2d", "scale", 2.5, 2.5);
    roundTrip ("transform2d", "translateX", -0.3, -0.3);
    roundTrip ("transform2d", "translateY", 0.42, 0.42);
    roundTrip ("transform2d", "rotation", 95.0, 95.0);
    roundTrip ("transform2d", "cropLeft", 0.25, 0.25);
    roundTrip ("transform2d", "cropRight", 1.7, 1.0);    // clamped high
    roundTrip ("transform2d", "cropTop", -0.5, 0.0);     // clamped low
    roundTrip ("transform2d", "cropBottom", 0.9, 0.9);

    // source — opacity clamps; visible is a >=0.5 threshold; zOrder/blendMode pass.
    roundTrip ("source", "opacity", 0.5, 0.5);
    roundTrip ("source", "opacity", 1.9, 1.0);           // clamped high
    roundTrip ("source", "opacity", -0.2, 0.0);          // clamped low
    roundTrip ("source", "visible", 0.7, 1.0);           // >= 0.5 -> visible
    roundTrip ("source", "visible", 0.3, 0.0);           // <  0.5 -> hidden
    roundTrip ("source", "zOrder", 3.0, 3.0);
    roundTrip ("source", "blendMode", 2.0, 2.0);

    // mask — type clamps to [0,2]; w/h clamp to >=0; feather to [0,1]; invert threshold.
    roundTrip ("mask", "type", 5.0, 2.0);                // clamped to 2
    roundTrip ("mask", "cx", 0.3, 0.3);
    roundTrip ("mask", "w", -1.0, 0.0);                  // clamped to >= 0
    roundTrip ("mask", "feather", 2.0, 1.0);             // clamped to 1
    roundTrip ("mask", "invert", 0.9, 1.0);              // >= 0.5 -> inverted

    // effect slot — setting the type, then a param, clamped to the effect's table range.
    {
        const int    type0   = 0;                                  // first registered effect
        const auto&  def      = videofx::kEffectDefs[type0];
        const char*  pName    = def.params[0].name;                // table-driven (robust to edits)
        const double pMin     = def.params[0].minValue;
        const double pMax     = def.params[0].maxValue;
        const double pDefault = def.params[0].defaultValue;

        MockClip c;
        check (videoparam::applyGraphParam (c, "effect0", "type", (double) type0), "apply effect0/type");
        // Setting the type seeds the param table defaults.
        double readDefault = -999.0;
        check (videoparam::getGraphParam (c, "effect0", pName, readDefault), "get effect0 default param");
        checkNear (readDefault, pDefault, 1e-5, "effect0 type seeds table default");

        // enabled is a >= 0.5 bool.
        check (videoparam::applyGraphParam (c, "effect0", "enabled", 1.0), "apply effect0/enabled");
        double en = -1.0;
        videoparam::getGraphParam (c, "effect0", "enabled", en);
        checkNear (en, 1.0, 1e-9, "effect0 enabled reads back true");

        // A param above the table max clamps to max; below min clamps to min.
        videoparam::applyGraphParam (c, "effect0", pName, pMax + 100.0);
        double hi = -1.0; videoparam::getGraphParam (c, "effect0", pName, hi);
        checkNear (hi, pMax, 1e-4, "effect param clamps to table max");
        videoparam::applyGraphParam (c, "effect0", pName, pMin - 100.0);
        double lo = -1.0; videoparam::getGraphParam (c, "effect0", pName, lo);
        checkNear (lo, pMin, 1e-4, "effect param clamps to table min");

        // An out-of-range slot index is rejected (not a silent write).
        check (! videoparam::applyGraphParam (c, "effect999", "type", 0.0), "out-of-range effect slot rejected");
    }

    // Unknown node / param -> false, so a caller's own namespace (e.g. exporter "gen")
    // gets its turn instead of being swallowed.
    {
        MockClip c;
        check (! videoparam::applyGraphParam (c, "gen", "uIntensity", 0.5), "unknown node 'gen' -> false (caller glue)");
        check (! videoparam::applyGraphParam (c, "source", "bogusParam", 1.0), "unknown param -> false");
        double out = 0.0;
        check (! videoparam::getGraphParam (c, "transform2d", "bogusParam", out), "unknown get param -> false");
    }

    std::printf ("video-param-grammar: %d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail;
}
