// effect_defs.h — canonical video effect definitions shared by the Arbit
// plugin and arbit-video-helper. Dependency-free (no JUCE, no libav, no GL)
// so it can be included from both sides of the GPL boundary; this header is
// plain data, original to this project.
//
// The table mirrors the reference implementation's EFFECT_DEFAULTS /
// EFFECT_BITS (bpm-sync-editor compositor/effects.py) exactly: same effect
// order, same shader bit values, same parameter names/defaults/ranges. The
// plugin uses it for the effects-rack UI, automation lane naming and
// (de)normalization; the helper uses it for uniform upload and
// graph_set_param routing. Do not reorder — enum order, bit values and
// per-effect parameter order are wire/serialization contracts.
#pragma once

namespace videofx
{

enum class EffectType : int
{
    Brightness = 0,
    Contrast,
    Saturation,
    Hue,
    Exposure,
    Gamma,
    Blur,
    Sharpen,
    Vignette,
    Warm,
    Cool,
    Vintage,
    Sepia,
    BlackAndWhite,
    Invert,
    Posterize,
    Noise,
    // ---- Arbit color-suite additions (Jun 2026). APPEND-ONLY: enum order,
    // bits and param order below are wire/serialization contracts.
    ChromaKey,
    LumaKey,
    ColorWheels,
    // ---- Geometric / UV effect passes (Jun 2026). APPEND-ONLY. These are
    // per-pass UV distortions, realized as separate ping-pong passes in the
    // renderer (NOT branches in the color uber-shader — the geometric family
    // stays per-pass, master-plan principle). The plugin treats them like any
    // other rack effect: table-driven menu, serialization and automation all
    // pick them up with no code change.
    Kaleidoscope,
    Mirror,
    Tile,
    Warp,
    Displace,
    // FeedbackTrail is the cross-frame geometric pass: it reads the previous
    // frame's output from a persistent per-(clip,slot) history FBO, so the
    // renderer handles it separately from the stateless geometric passes.
    FeedbackTrail,
    // More stateless geometric passes (appended after FeedbackTrail to keep
    // enum values stable for serialization; they need not be contiguous with
    // Kaleidoscope..Displace — the renderer keys them by an explicit table).
    PolarSwirl,
    DisplaceRgb,
    Pixelate,
    Count
};

static constexpr int kEffectTypeCount = static_cast<int>(EffectType::Count);

// Shader bitmask values (uEffectMask), identical to the reference editor.
// Blur/Sharpen have bits reserved but are realized as a separate
// neighborhood-sampling pass, not branches in the color uber-shader.
static constexpr int kEffectBits[kEffectTypeCount] = {
    1,      // Brightness
    2,      // Contrast
    4,      // Saturation
    8,      // Hue
    16,     // Exposure
    32,     // Gamma
    64,     // Blur
    128,    // Sharpen
    256,    // Vignette
    512,    // Warm
    1024,   // Cool
    2048,   // Vintage
    4096,   // Sepia
    8192,   // BlackAndWhite
    16384,  // Invert
    32768,  // Posterize
    65536,  // Noise
    131072, // ChromaKey
    262144, // LumaKey
    524288, // ColorWheels
    1048576,  // Kaleidoscope
    2097152,  // Mirror
    4194304,  // Tile
    8388608,  // Warp
    16777216, // Displace
    33554432, // FeedbackTrail
    67108864,  // PolarSwirl
    134217728, // DisplaceRgb
    268435456, // Pixelate
};

// Max parameters any single effect exposes (ColorWheels has 9: lift/gamma/
// gain x RGB). VideoEffectInstance stores a fixed array of this size and
// automation paramIdx addresses into it, so this is also a wire contract.
// Bumped 4 -> 9 (Jun 2026, color suite): safe because every consumer
// (StateSerializer p0..pN attributes, MidiExporter token lists, jobSpec /
// graph_set_effects name->value maps, the effects-rack UI and the automation
// subParam packing slot*100 + paramIdx with paramIdx < 99) is count-driven,
// and old serialized data simply stops at p3.
static constexpr int kMaxEffectParams = 9;

struct EffectParamDef
{
    const char* name;         // wire name used in graph param IDs ("radius")
    const char* displayName;  // UI label ("Blur Radius")
    float defaultValue;
    float minValue;
    float maxValue;
    float step;
    const char* unit;         // "" when unitless
};

struct EffectDef
{
    EffectType type;
    const char* name;         // wire name used in serialization ("blur")
    const char* displayName;  // UI label ("Blur")
    const char* category;     // rack "+ Add" menu grouping
    int bit;                  // uEffectMask bit
    int paramCount;
    EffectParamDef params[kMaxEffectParams];
};

static constexpr EffectDef kEffectDefs[kEffectTypeCount] = {
    { EffectType::Brightness, "brightness", "Brightness", "Color Adjustments", 1,
      1, { { "amount", "Brightness", 0.0f, -1.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Contrast, "contrast", "Contrast", "Color Adjustments", 2,
      1, { { "amount", "Contrast", 1.0f, 0.0f, 3.0f, 0.01f, "" } } },
    { EffectType::Saturation, "saturation", "Saturation", "Color Adjustments", 4,
      1, { { "amount", "Saturation", 1.0f, 0.0f, 3.0f, 0.01f, "" } } },
    { EffectType::Hue, "hue", "Hue Shift", "Color Adjustments", 8,
      1, { { "shift", "Hue Shift", 0.0f, -180.0f, 180.0f, 1.0f, "deg" } } },
    { EffectType::Exposure, "exposure", "Exposure", "Color Adjustments", 16,
      1, { { "stops", "Exposure", 0.0f, -4.0f, 4.0f, 0.1f, "stops" } } },
    { EffectType::Gamma, "gamma", "Gamma", "Color Adjustments", 32,
      1, { { "value", "Gamma", 1.0f, 0.1f, 3.0f, 0.01f, "" } } },
    { EffectType::Blur, "blur", "Blur", "Filters", 64,
      1, { { "radius", "Blur Radius", 0.0f, 0.0f, 20.0f, 0.5f, "px" } } },
    { EffectType::Sharpen, "sharpen", "Sharpen", "Filters", 128,
      1, { { "amount", "Sharpen", 0.0f, 0.0f, 5.0f, 0.1f, "" } } },
    { EffectType::Vignette, "vignette", "Vignette", "Filters", 256,
      2, { { "amount", "Amount", 0.0f, 0.0f, 1.0f, 0.01f, "" },
           { "softness", "Softness", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Warm, "warm", "Warm", "Color Grading", 512,
      1, { { "intensity", "Intensity", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Cool, "cool", "Cool", "Color Grading", 1024,
      1, { { "intensity", "Intensity", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Vintage, "vintage", "Vintage", "Color Grading", 2048,
      1, { { "intensity", "Intensity", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Sepia, "sepia", "Sepia", "Color Grading", 4096,
      1, { { "intensity", "Intensity", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::BlackAndWhite, "black_and_white", "Black & White", "Color Grading", 8192,
      1, { { "intensity", "Intensity", 1.0f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Invert, "invert", "Invert", "Stylize", 16384,
      1, { { "amount", "Amount", 1.0f, 0.0f, 1.0f, 0.01f, "" } } },
    { EffectType::Posterize, "posterize", "Posterize", "Stylize", 32768,
      1, { { "levels", "Color Levels", 8.0f, 2.0f, 32.0f, 1.0f, "" } } },
    { EffectType::Noise, "noise", "Film Grain", "Stylize", 65536,
      1, { { "amount", "Amount", 0.1f, 0.0f, 1.0f, 0.01f, "" } } },
    // ---- Color suite (Jun 2026). Append-only.
    // Chroma key: alpha = smoothstep over chroma distance (CbCr plane) from
    // the key color; spillSuppress desaturates the keyed hue in kept pixels.
    // Keying runs FIRST in the uber-shader so the produced alpha feeds the
    // blend/composite pass.
    { EffectType::ChromaKey, "chroma_key", "Chroma Key", "Keying", 131072,
      6, { { "keyR", "Key Red", 0.0f, 0.0f, 1.0f, 0.01f, "" },
           { "keyG", "Key Green", 1.0f, 0.0f, 1.0f, 0.01f, "" },
           { "keyB", "Key Blue", 0.0f, 0.0f, 1.0f, 0.01f, "" },
           { "tolerance", "Tolerance", 0.18f, 0.0f, 1.0f, 0.005f, "" },
           { "softness", "Softness", 0.10f, 0.0f, 1.0f, 0.005f, "" },
           { "spillSuppress", "Spill Suppress", 0.5f, 0.0f, 1.0f, 0.01f, "" } } },
    // Luma key: keeps luma in [low, high] (alpha 1), feathered outward by
    // softness; invert >= 0.5 keeps the outside instead. Defaults keep all.
    { EffectType::LumaKey, "luma_key", "Luma Key", "Keying", 262144,
      4, { { "low", "Low", 0.0f, 0.0f, 1.0f, 0.005f, "" },
           { "high", "High", 1.0f, 0.0f, 1.0f, 0.005f, "" },
           { "softness", "Softness", 0.1f, 0.0f, 1.0f, 0.005f, "" },
           { "invert", "Invert", 0.0f, 0.0f, 1.0f, 1.0f, "" } } },
    // Lift/gamma/gain color wheels (per-channel):
    // out = pow(clamp(in * gain + lift * (1 - in)), 1 / gamma).
    { EffectType::ColorWheels, "color_wheels", "Color Wheels", "Color Grading", 524288,
      9, { { "liftR", "Lift R", 0.0f, -1.0f, 1.0f, 0.005f, "" },
           { "liftG", "Lift G", 0.0f, -1.0f, 1.0f, 0.005f, "" },
           { "liftB", "Lift B", 0.0f, -1.0f, 1.0f, 0.005f, "" },
           { "gammaR", "Gamma R", 1.0f, 0.2f, 4.0f, 0.01f, "" },
           { "gammaG", "Gamma G", 1.0f, 0.2f, 4.0f, 0.01f, "" },
           { "gammaB", "Gamma B", 1.0f, 0.2f, 4.0f, 0.01f, "" },
           { "gainR", "Gain R", 1.0f, 0.0f, 4.0f, 0.01f, "" },
           { "gainG", "Gain G", 1.0f, 0.0f, 4.0f, 0.01f, "" },
           { "gainB", "Gain B", 1.0f, 0.0f, 4.0f, 0.01f, "" } } },
    // ---- Geometric / UV effect passes (Jun 2026). Append-only. Realized as
    // per-slot ping-pong passes in the renderer (kKaleidoscopeFragment etc.),
    // each reading the previous result; deliberately NOT branches in the color
    // uber-shader. The shader's float uniform names equal these wire param
    // names, so init can fetch locations straight from this table.
    { EffectType::Kaleidoscope, "kaleidoscope", "Kaleidoscope", "Geometry", 1048576,
      4, { { "segments", "Segments", 6.0f, 2.0f, 24.0f, 1.0f, "" },
           { "angle", "Angle", 0.0f, 0.0f, 1.0f, 0.01f, "turns" },
           { "spin", "Spin", 0.0f, -1.0f, 1.0f, 0.01f, "turns/s" },
           { "zoom", "Zoom", 1.0f, 0.2f, 4.0f, 0.01f, "" } } },
    { EffectType::Mirror, "mirror", "Mirror", "Geometry", 2097152,
      1, { { "mode", "Mode", 0.0f, 0.0f, 3.0f, 1.0f, "" } } },
    { EffectType::Tile, "tile", "Tile", "Geometry", 4194304,
      2, { { "count", "Tiles", 2.0f, 1.0f, 8.0f, 1.0f, "" },
           { "mirrorEdges", "Mirror Edges", 1.0f, 0.0f, 1.0f, 1.0f, "" } } },
    { EffectType::Warp, "warp", "Warp", "Geometry", 8388608,
      3, { { "amount", "Amount", 0.03f, 0.0f, 0.25f, 0.005f, "" },
           { "frequency", "Frequency", 8.0f, 1.0f, 40.0f, 0.5f, "" },
           { "speed", "Speed", 1.0f, 0.0f, 8.0f, 0.1f, "" } } },
    { EffectType::Displace, "displace", "Displace", "Geometry", 16777216,
      1, { { "amount", "Amount", 0.02f, 0.0f, 0.2f, 0.005f, "" } } },
    // Cross-frame: blends the source over a decayed, zoomed/rotated copy of the
    // previous frame's output (persistent history FBO). decay = trail length,
    // zoom/swirl = how the trail drifts each frame. Run last, <=1 per clip.
    { EffectType::FeedbackTrail, "feedback_trail", "Feedback Trail", "Geometry", 33554432,
      3, { { "decay", "Decay", 0.92f, 0.0f, 0.999f, 0.005f, "" },
           { "zoom", "Zoom", 0.99f, 0.9f, 1.05f, 0.001f, "" },
           { "swirl", "Swirl", 0.0f, -0.1f, 0.1f, 0.002f, "rad" } } },
    // Localized vortex: rotation peaks at the centre, fading to none past
    // `radius`. amount is in turns at the centre.
    { EffectType::PolarSwirl, "polar_swirl", "Polar Swirl", "Geometry", 67108864,
      2, { { "amount", "Amount", 0.5f, -2.0f, 2.0f, 0.05f, "turn" },
           { "radius", "Radius", 0.7f, 0.05f, 1.5f, 0.05f, "" } } },
    // Chromatic shift: R/B sampled from opposite offsets along `angle`.
    { EffectType::DisplaceRgb, "displace_rgb", "RGB Shift", "Geometry", 134217728,
      2, { { "amount", "Amount", 0.01f, 0.0f, 0.1f, 0.002f, "" },
           { "angle", "Angle", 0.0f, 0.0f, 1.0f, 0.01f, "turn" } } },
    // Mosaic: snap UVs to a coarse grid; `size` is the block edge in px.
    { EffectType::Pixelate, "pixelate", "Pixelate", "Geometry", 268435456,
      1, { { "size", "Size", 8.0f, 1.0f, 128.0f, 1.0f, "px" } } },
};

inline const EffectDef* effectDefFor(int type)
{
    if (type < 0 || type >= kEffectTypeCount)
        return nullptr;
    return &kEffectDefs[type];
}

// Find an effect's param index by wire name; -1 if absent.
inline int effectParamIndex(int type, const char* paramName)
{
    const EffectDef* def = effectDefFor(type);
    if (def == nullptr || paramName == nullptr)
        return -1;
    for (int i = 0; i < def->paramCount; ++i)
    {
        const char* a = def->params[i].name;
        const char* b = paramName;
        while (*a != '\0' && *a == *b) { ++a; ++b; }
        if (*a == '\0' && *b == '\0')
            return i;
    }
    return -1;
}

// Blend modes for the composite pass — values match the reference editor's
// BlendMode enum and the existing clip<id>/source/blendMode wire values.
enum class BlendMode : int
{
    Normal = 0,
    Add,
    Multiply,
    Screen,
    Overlay,
    Count
};

static constexpr const char* kBlendModeNames[static_cast<int>(BlendMode::Count)] = {
    "Normal", "Add", "Multiply", "Screen", "Overlay",
};

// Transition types — values match the reference editor's TransitionType enum
// order (none, fade, dissolve, wipe left/right/up/down).
enum class TransitionType : int
{
    None = 0,
    Fade,
    Dissolve,
    WipeLeft,
    WipeRight,
    WipeUp,
    WipeDown,
    Count
};

static constexpr const char* kTransitionNames[static_cast<int>(TransitionType::Count)] = {
    "None", "Fade", "Dissolve", "Wipe Left", "Wipe Right", "Wipe Up", "Wipe Down",
};

// Max effect rack slots per clip. Automation lane addressing packs the slot
// index into the lane subParam, so this is a serialization contract too.
static constexpr int kMaxEffectSlots = 8;

} // namespace videofx
