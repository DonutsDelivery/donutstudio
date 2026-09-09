// renderer.cpp — FrameRenderer implementation. Shader sources are verbatim
// ports from the reference editor (artifacts/video-parity/*.md); deviations
// are marked "Arbit extension" or "conscious fix" with the spec reference.

#if ARBIT_HAVE_VIEWPORT

#include "renderer.h"
#include "visual_renderer_telemetry.h"
#include "visual_plan_executor.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace videorender
{

FrameRenderer::~FrameRenderer()
{
    shutdown();
}

struct FrameRenderer::FramePublicationCandidate
{
    FrameRenderer* renderer = nullptr;
    std::uint64_t owner = 0;
    std::uint64_t generation = 0;
    std::uint64_t serial = 0;
    std::uint64_t basePublicationGeneration = 0;
    std::shared_ptr<FramePublicationCandidate> predecessor;
    bool committed = false;
    bool discarded = false;
    std::map<int, FeedbackHistory> feedbackHistory;
    std::map<std::pair<int, int>, FrameDelayHistory> frameDelayHistory;
    std::uint64_t frameDelayHistorySerial = 0;
    unsigned frameParity = 0;
    videowire::RenderPassOutputPublicationPtr renderPassOutputs;
    std::function<void(FramePublicationCandidate&)> release;
    ~FramePublicationCandidate() { if (release) release(*this); }
};

namespace
{

uint64_t fnv1a (const uint8_t* data, size_t n)
{
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < n; ++i)
    {
        h ^= data[i];
        h *= 1099511628211ull;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Shared vertex shader — verbatim (compositor-spec §3.1). Used by every pass;
// non-geometry passes set uTransform = identity and uCrop = 0 explicitly
// (the reference relied on driver-zeroed uniforms — see the §3.1 porting
// warning).
const char* kVertexShader = R"GLSL(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;

uniform mat4 uTransform;
uniform vec4 uCrop;  // left, right, top, bottom
uniform int uCornerPin;
uniform vec2 uCorners[4];

out vec2 TexCoord;
out vec2 RawUV;      // un-cropped quad UV (Arbit extension: shape mask space)

void main() {
    vec4 affinePosition = uTransform * vec4(aPos, 0.0, 1.0);
    vec2 top = mix(uCorners[0], uCorners[1], aTexCoord.x);
    vec2 bottom = mix(uCorners[3], uCorners[2], aTexCoord.x);
    gl_Position = uCornerPin != 0 ? vec4(mix(top, bottom, aTexCoord.y), 0.0, 1.0)
                                  : affinePosition;

    // Apply crop by remapping texture coordinates
    vec2 cropMin = vec2(uCrop.x, uCrop.z);
    vec2 cropMax = vec2(1.0 - uCrop.y, 1.0 - uCrop.w);
    TexCoord = mix(cropMin, cropMax, aTexCoord);
    RawUV = aTexCoord;
}
)GLSL";

// Layer fragment shader — verbatim core (compositor-spec §5.1; uBlendMode is
// declared-but-unused in the reference too) plus the Arbit shape-mask
// extension: a rect/ellipse mask in the clip's displayed-frame UV space
// (RawUV: 0..1, origin top-left because texture row 0 = image top) multiplies
// the layer alpha, so the keyed/masked alpha feeds the blend pass.
const char* kLayerFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
in vec2 RawUV;
out vec4 FragColor;

uniform sampler2D uTexture;
uniform float uOpacity;
uniform int uBlendMode;  // 0=normal, 1=add, 2=multiply, 3=screen, 4=overlay
uniform int uMaskType;       // 0 none, 1 rect, 2 ellipse
uniform vec4 uMaskRect;      // cx, cy, w, h (full extents)
uniform float uMaskFeather;  // feather width in mask UV units
uniform int uMaskInvert;
uniform vec4 uPathRect;      // cx, cy, signed width (ellipse < 0), height
uniform vec4 uPathRectB;
uniform int uPathOperation;  // -1 disabled, 0 first, 1 add, 2 intersect, 3 subtract
uniform int uPathInvert;
uniform sampler2D uMatteTexture;
uniform sampler2D uMatteTextureB;
uniform int uMatteCombineMode;
uniform int uMatteApply;
uniform vec2 uMatteTexel;
uniform vec4 uMatteRefine; // black, white, erode/dilate px, feather px
uniform float uMatteChoke;
uniform int uMatteInvert;
uniform sampler2D uDepthTexture;
uniform int uDepthFog;
uniform vec3 uFogRangeDensity;
uniform vec4 uFogColor;

float rawMatteCoverage(vec2 uv) {
    float value = texture(uMatteTexture, uv).r;
    if (uMatteCombineMode >= 0) {
        float b = texture(uMatteTextureB, uv).r;
        if (uMatteCombineMode == 0) value = max(value, b);
        else if (uMatteCombineMode == 1) value = min(value, b);
        else if (uMatteCombineMode == 2) value = max(value - b, 0.0);
        else value = abs(value - b);
    }
    return value;
}
float matteCoverage(vec2 uv) {
    if (uMatteApply == 0) return 1.0;
    int radius = int(clamp(abs(uMatteRefine.z), 0.0, 4.0));
    float value = rawMatteCoverage(uv);
    if (radius > 0) {
        float aggregate = uMatteRefine.z >= 0.0 ? 0.0 : 1.0;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= radius && abs(y) <= radius) {
                float sampleValue = rawMatteCoverage(uv + vec2(x,y) * uMatteTexel);
                aggregate = uMatteRefine.z >= 0.0 ? max(aggregate, sampleValue)
                                                  : min(aggregate, sampleValue);
            }
        value = aggregate;
    }
    int feather = int(clamp(uMatteRefine.w, 0.0, 4.0));
    if (feather > 0) {
        float sum = 0.0, count = 0.0;
        for (int y = -4; y <= 4; ++y) for (int x = -4; x <= 4; ++x)
            if (abs(x) <= feather && abs(y) <= feather) {
                sum += rawMatteCoverage(uv + vec2(x,y) * uMatteTexel); count += 1.0;
            }
        value = sum / max(count, 1.0);
    }
    value = clamp((value - uMatteRefine.x) / max(uMatteRefine.y - uMatteRefine.x, 1e-5), 0.0, 1.0);
    value = clamp(value + uMatteChoke, 0.0, 1.0);
    return uMatteInvert != 0 ? 1.0 - value : value;
}

float maskCoverage(vec2 uv) {
    if (uMaskType == 0) return 1.0;
    float f = max(uMaskFeather, 1e-5);
    float cov;
    if (uMaskType == 1) {
        // Rect: signed distance inside each half-extent, feathered inward.
        vec2 d = abs(uv - uMaskRect.xy) - 0.5 * uMaskRect.zw;
        float dist = max(d.x, d.y);             // > 0 outside
        cov = 1.0 - smoothstep(-f, 0.0, dist);
    } else {
        // Ellipse: normalized radius, feathered around r = 1.
        vec2 r = (uv - uMaskRect.xy) / max(0.5 * uMaskRect.zw, vec2(1e-5));
        float dist = length(r) - 1.0;
        // Feather expressed in mean-extent units so it visually matches rect.
        float fr = f / max(0.25 * (uMaskRect.z + uMaskRect.w), 1e-5);
        cov = 1.0 - smoothstep(-fr, 0.0, dist);
    }
    return uMaskInvert != 0 ? 1.0 - cov : cov;
}

float pathShapeCoverage(vec2 uv, vec4 shape) {
    if (shape.z < 0.0) {
        vec2 r = (uv - shape.xy) / max(0.5 * vec2(-shape.z, shape.w), vec2(1e-5));
        return step(length(r), 1.0);
    }
    vec2 d = abs(uv - shape.xy) - 0.5 * shape.zw;
    return step(max(d.x, d.y), 0.0);
}

float pathCoverage(vec2 uv) {
    if (uPathOperation < 0) return 1.0;
    float a = pathShapeCoverage(uv, uPathRect);
    float value = a;
    if (uPathOperation > 0) {
        float b = pathShapeCoverage(uv, uPathRectB);
        if (uPathOperation == 1) value = max(a, b);
        else if (uPathOperation == 2) value = min(a, b);
        else value = max(a - b, 0.0);
    }
    return uPathInvert != 0 ? 1.0 - value : value;
}

void main() {
    float depth = texture(uDepthTexture, RawUV).r;
    vec2 sampleUV = TexCoord;
    if (uDepthFog == 3)
        sampleUV = clamp(TexCoord + vec2(uFogRangeDensity.x, uFogRangeDensity.y)
            * (depth - uFogRangeDensity.z), vec2(0.0), vec2(1.0));
    vec4 texColor = texture(uTexture, sampleUV);
    if (uDepthFog == 2) {
        float blur = clamp(abs(depth - uFogRangeDensity.y) / max(uFogRangeDensity.z, 1e-5), 0.0, 1.0)
            * uFogRangeDensity.x;
        vec2 stepUV = blur / vec2(textureSize(uTexture, 0));
        texColor = (texColor * 4.0 + texture(uTexture, sampleUV + vec2(stepUV.x, 0.0))
            + texture(uTexture, sampleUV - vec2(stepUV.x, 0.0))
            + texture(uTexture, sampleUV + vec2(0.0, stepUV.y))
            + texture(uTexture, sampleUV - vec2(0.0, stepUV.y))) / 8.0;
    }
    vec3 rgb = texColor.rgb;
    if (uDepthFog == 1) {
        float range = clamp((depth - uFogRangeDensity.x) /
                            max(uFogRangeDensity.y - uFogRangeDensity.x, 1e-5), 0.0, 1.0);
        float amount = uFogColor.a * (1.0 - exp(-uFogRangeDensity.z * range));
        rgb = mix(rgb, uFogColor.rgb, amount);
    } else if (uDepthFog == 4) {
        rgb *= uFogColor.rgb * max(0.0, uFogRangeDensity.y
            + uFogRangeDensity.x * (1.0 - depth));
    }
    FragColor = vec4(rgb, texColor.a * uOpacity * maskCoverage(RawUV)
        * pathCoverage(RawUV) * matteCoverage(RawUV));
}
)GLSL";

// Draw Shape image producer. Geometry is expressed in normalized frame UVs,
// matching the native Metal compositor exactly; negative widths are the private
// rectangle/ellipse discriminators. Operation 0 draws A, 1 unions, 2 intersects,
// and 3 subtracts B from A. All colour channels remain straight RGBA.
const char* kDrawShapeFragment = R"GLSL(#version 330 core
in vec2 RawUV;
out vec4 FragColor;

uniform vec4 uShapeRect;   // cx, cy, signed width (ellipse < 0), height
uniform vec4 uShapeRectB;
uniform int uShapeOperation;
uniform vec4 uShapeColor;

bool shapeContains(vec4 rect) {
    vec2 halfSize = max(abs(rect.zw) * 0.5, vec2(0.0));
    vec2 local = abs(RawUV - rect.xy);
    bool inside;
    if (rect.z < 0.0) {
        if (any(lessThanEqual(halfSize, vec2(0.0))))
            inside = false;
        else {
            vec2 normalized = local / halfSize;
            inside = dot(normalized, normalized) <= 1.0;
        }
    } else {
        vec2 d = local - halfSize;
        inside = max(d.x, d.y) <= 0.0;
    }
    return inside;
}

void main() {
    bool insideA = shapeContains(uShapeRect);
    bool insideB = uShapeOperation == 0 ? false : shapeContains(uShapeRectB);
    bool inside = uShapeOperation == 1 ? (insideA || insideB)
                : uShapeOperation == 2 ? (insideA && insideB)
                : uShapeOperation == 3 ? (insideA && !insideB)
                : insideA;
    FragColor = inside ? uShapeColor : vec4(0.0);
}
)GLSL";

// Blend fragment shader (compositor-spec §5.2 plus the extended editor modes).
const char* kBlendFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uFrontTex;
uniform sampler2D uBackTex;
uniform float uOpacity;
uniform int uBlendMode;

vec3 blendOverlay(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        result[i] = base[i] < 0.5
            ? 2.0 * base[i] * blend[i]
            : 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - blend[i]);
    }
    return result;
}

vec3 blendColorDodge(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (base[i] <= 0.0) result[i] = 0.0;
        else if (blend[i] >= 1.0) result[i] = 1.0;
        else result[i] = min(1.0, base[i] / (1.0 - blend[i]));
    }
    return result;
}

vec3 blendColorBurn(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (base[i] >= 1.0) result[i] = 1.0;
        else if (blend[i] <= 0.0) result[i] = 0.0;
        else result[i] = 1.0 - min(1.0, (1.0 - base[i]) / blend[i]);
    }
    return result;
}

vec3 blendSoftLight(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        if (blend[i] <= 0.5) {
            result[i] = base[i] - (1.0 - 2.0 * blend[i]) * base[i] * (1.0 - base[i]);
        } else {
            float d = base[i] <= 0.25
                ? ((16.0 * base[i] - 12.0) * base[i] + 4.0) * base[i]
                : sqrt(base[i]);
            result[i] = base[i] + (2.0 * blend[i] - 1.0) * (d - base[i]);
        }
    }
    return result;
}

vec3 blendHardLight(vec3 base, vec3 blend) {
    vec3 result;
    for (int i = 0; i < 3; i++) {
        result[i] = blend[i] < 0.5
            ? 2.0 * base[i] * blend[i]
            : 1.0 - 2.0 * (1.0 - base[i]) * (1.0 - blend[i]);
    }
    return result;
}

vec3 applyBlendMode(vec3 base, vec3 blend, int mode) {
    if (mode == 1) return min(base + blend, vec3(1.0));
    if (mode == 2) return base * blend;
    if (mode == 3) return 1.0 - (1.0 - base) * (1.0 - blend);
    if (mode == 4) return blendOverlay(base, blend);
    if (mode == 5) return abs(base - blend);
    if (mode == 6) return base + blend - 2.0 * base * blend;
    if (mode == 7) return min(base, blend);
    if (mode == 8) return max(base, blend);
    if (mode == 9) return blendColorDodge(base, blend);
    if (mode == 10) return blendColorBurn(base, blend);
    if (mode == 11) return blendSoftLight(base, blend);
    if (mode == 12) return blendHardLight(base, blend);
    return blend;
}

void main() {
    vec4 front = texture(uFrontTex, TexCoord);
    vec4 back = texture(uBackTex, TexCoord);
    float alpha = front.a * uOpacity;
    vec3 blended = applyBlendMode(back.rgb, front.rgb, uBlendMode);

    // Alpha compositing
    FragColor = vec4(mix(back.rgb, blended, alpha), max(back.a, alpha));
}
)GLSL";

// Transition fragment shader — verbatim (compositor-spec §5.5 /
// overlay-fx-spec §1.7). Shader type indices are the reference's
// (0=dissolve, 1=fade, 2..5=wipes); the wire enum (videofx::TransitionType)
// is mapped in transitionShaderType() below.
const char* kTransitionFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D uFromTex;   // Outgoing clip texture
uniform sampler2D uToTex;     // Incoming clip texture
uniform float uProgress;       // Transition progress (0.0 to 1.0)
uniform int uTransitionType;   // 0=dissolve, 1=fade, 2=wipe_left, 3=wipe_right, 4=wipe_up, 5=wipe_down

// Ease in-out cubic for smooth transitions
float easeInOutCubic(float t) {
    if (t < 0.5) {
        return 4.0 * t * t * t;
    } else {
        float p = 2.0 * t - 2.0;
        return 0.5 * p * p * p + 1.0;
    }
}

void main() {
    vec4 fromColor = texture(uFromTex, TexCoord);
    vec4 toColor = texture(uToTex, TexCoord);

    float progress = uProgress;

    if (uTransitionType == 0) {
        // Dissolve - cross-fade with easing
        float eased = easeInOutCubic(progress);
        FragColor = mix(fromColor, toColor, eased);
    }
    else if (uTransitionType == 1) {
        // Fade through black
        if (progress < 0.5) {
            // Fade out
            float fadeOut = 1.0 - progress * 2.0;
            FragColor = vec4(fromColor.rgb * fadeOut, fromColor.a);
        } else {
            // Fade in
            float fadeIn = (progress - 0.5) * 2.0;
            FragColor = vec4(toColor.rgb * fadeIn, toColor.a);
        }
    }
    else if (uTransitionType == 2) {
        // Wipe left (incoming from right)
        float edge = 1.0 - progress;
        if (TexCoord.x > edge) {
            FragColor = toColor;
        } else {
            FragColor = fromColor;
        }
    }
    else if (uTransitionType == 3) {
        // Wipe right (incoming from left)
        if (TexCoord.x < progress) {
            FragColor = toColor;
        } else {
            FragColor = fromColor;
        }
    }
    else if (uTransitionType == 4) {
        // Wipe up (incoming from bottom)
        float edge = 1.0 - progress;
        if (TexCoord.y > edge) {
            FragColor = toColor;
        } else {
            FragColor = fromColor;
        }
    }
    else if (uTransitionType == 5) {
        // Wipe down (incoming from top)
        if (TexCoord.y < progress) {
            FragColor = toColor;
        } else {
            FragColor = fromColor;
        }
    }
    else {
        // Default: instant cut at 50%
        FragColor = progress >= 0.5 ? toColor : fromColor;
    }
}
)GLSL";

// Effects uber-shader — verbatim port of the reference
// EffectsShaderLibrary.get_full_shader() = SHADER_HEADER + EFFECTS_MAIN
// (effects-spec §3.2). The EFF_* constants must equal videofx::kEffectBits;
// a static_assert below enforces it. Blur (bit 64) and Sharpen (bit 128)
// have no branch here, exactly like the reference — they are realized as
// separate neighborhood passes (Arbit extension, effects-spec §5).
const char* kEffectsFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float uTime;
uniform float uBrightness;
uniform float uContrast;
uniform float uSaturation;
uniform float uHueShift;
uniform float uExposure;
uniform float uGamma;
uniform float uVignetteAmount;
uniform float uVignetteSoftness;
uniform float uWarmth;
uniform float uCoolness;
uniform float uVintageIntensity;
uniform float uSepiaIntensity;
uniform float uBWIntensity;
uniform float uInvertAmount;
uniform float uPosterizeLevels;
uniform float uNoiseAmount;
uniform float uKeyR;
uniform float uKeyG;
uniform float uKeyB;
uniform float uKeyTolerance;
uniform float uKeySoftness;
uniform float uKeySpill;
uniform float uKeyChoke;
uniform float uKeyFeather;
uniform float uKeyEdgeR;
uniform float uKeyEdgeG;
uniform float uKeyEdgeB;
uniform float uKeyEdgeAmount;
uniform float uKeyMatteView;
uniform float uKeyTexelX;
uniform float uKeyTexelY;
uniform float uLumaLow;
uniform float uLumaHigh;
uniform float uLumaSoftness;
uniform float uLumaInvert;
uniform float uLiftR;
uniform float uLiftG;
uniform float uLiftB;
uniform float uGammaR;
uniform float uGammaG;
uniform float uGammaB;
uniform float uGainR;
uniform float uGainG;
uniform float uGainB;
uniform int uEffectMask;
const int EFF_BRIGHTNESS = 1;
const int EFF_CONTRAST = 2;
const int EFF_SATURATION = 4;
const int EFF_HUE = 8;
const int EFF_EXPOSURE = 16;
const int EFF_GAMMA = 32;
const int EFF_VIGNETTE = 256;
const int EFF_WARM = 512;
const int EFF_COOL = 1024;
const int EFF_VINTAGE = 2048;
const int EFF_SEPIA = 4096;
const int EFF_BW = 8192;
const int EFF_INVERT = 16384;
const int EFF_POSTERIZE = 32768;
const int EFF_NOISE = 65536;
const int EFF_CHROMA_KEY = 131072;
const int EFF_LUMA_KEY = 262144;
const int EFF_WHEELS = 524288;
vec3 rgb2hsl(vec3 c) { float maxC = max(max(c.r, c.g), c.b); float minC = min(min(c.r, c.g), c.b); float l = (maxC + minC) / 2.0; float h = 0.0, s = 0.0; if (maxC != minC) { float d = maxC - minC; s = l > 0.5 ? d / (2.0 - maxC - minC) : d / (maxC + minC); if (maxC == c.r) h = (c.g - c.b) / d + (c.g < c.b ? 6.0 : 0.0); else if (maxC == c.g) h = (c.b - c.r) / d + 2.0; else h = (c.r - c.g) / d + 4.0; h /= 6.0; } return vec3(h, s, l); }
float hue2rgb(float p, float q, float t) { if (t < 0.0) t += 1.0; if (t > 1.0) t -= 1.0; if (t < 1.0/6.0) return p + (q - p) * 6.0 * t; if (t < 0.5) return q; if (t < 2.0/3.0) return p + (q - p) * (2.0/3.0 - t) * 6.0; return p; }
vec3 hsl2rgb(vec3 hsl) { float h = hsl.x, s = hsl.y, l = hsl.z; vec3 rgb; if (s == 0.0) { rgb = vec3(l); } else { float q = l < 0.5 ? l * (1.0 + s) : l + s - l * s; float p = 2.0 * l - q; rgb.r = hue2rgb(p, q, h + 1.0/3.0); rgb.g = hue2rgb(p, q, h); rgb.b = hue2rgb(p, q, h - 1.0/3.0); } return rgb; }
float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float random(vec2 st) { return fract(sin(dot(st.xy, vec2(12.9898, 78.233))) * 43758.5453); }
float chromaCoverage(vec2 uv, vec3 key) {
    vec4 sampleColor = texture(uTexture, clamp(uv, 0.0, 1.0));
    vec2 pixCbCr = vec2(sampleColor.b - luminance(sampleColor.rgb),
                        sampleColor.r - luminance(sampleColor.rgb));
    vec2 keyCbCr = vec2(key.b - luminance(key), key.r - luminance(key));
    float dist = distance(pixCbCr, keyCbCr);
    return sampleColor.a * smoothstep(uKeyTolerance,
        uKeyTolerance + max(uKeySoftness, 1e-5), dist);
}
void main() {
    vec4 color = texture(uTexture, TexCoord);
    vec3 rgb = color.rgb;
    float alpha = color.a;
    // ---- Keying (Arbit color suite) runs FIRST, on the source colors, so
    // the produced alpha feeds every later pass and the blend/composite.
    if ((uEffectMask & EFF_CHROMA_KEY) != 0) {
        // Chroma distance in the CbCr plane (BT.709), luma-independent.
        vec3 key = vec3(uKeyR, uKeyG, uKeyB);
        float keyAlpha = chromaCoverage(TexCoord, key);
        if (uKeyFeather > 0.0) {
            int radius = int(ceil(min(uKeyFeather, 4.0)));
            float sigma = max(uKeyFeather * 0.5, 0.5);
            float sum = 0.0, weightSum = 0.0;
            for (int y = -4; y <= 4; ++y)
                for (int x = -4; x <= 4; ++x) {
                    if (abs(x) > radius || abs(y) > radius) continue;
                    float weight = exp(-float(x*x + y*y) / (2.0 * sigma * sigma));
                    sum += chromaCoverage(TexCoord + vec2(float(x) * uKeyTexelX,
                                                          float(y) * uKeyTexelY), key) * weight;
                    weightSum += weight;
                }
            keyAlpha = sum / max(weightSum, 1e-5);
        }
        alpha = clamp(keyAlpha + uKeyChoke, 0.0, 1.0);
        // Spill suppression: clamp the dominant key channel toward the other
        // two in kept pixels.
        if (uKeySpill > 0.0) {
            if (key.g >= key.r && key.g >= key.b) {
                float limit = max(rgb.r, rgb.b);
                rgb.g = mix(rgb.g, min(rgb.g, limit), uKeySpill);
            } else if (key.b >= key.r) {
                float limit = max(rgb.r, rgb.g);
                rgb.b = mix(rgb.b, min(rgb.b, limit), uKeySpill);
            } else {
                float limit = max(rgb.g, rgb.b);
                rgb.r = mix(rgb.r, min(rgb.r, limit), uKeySpill);
            }
        }
        float edgeWeight = uKeyEdgeAmount * 4.0 * alpha * (1.0 - alpha);
        rgb = mix(rgb, vec3(uKeyEdgeR, uKeyEdgeG, uKeyEdgeB), edgeWeight);
        if (uKeyMatteView >= 0.5) {
            FragColor = vec4(vec3(alpha), 1.0);
            return;
        }
    }
    if ((uEffectMask & EFF_LUMA_KEY) != 0) {
        float luma = luminance(rgb);
        float s = max(uLumaSoftness, 1e-5);
        float keep = smoothstep(uLumaLow - s, uLumaLow, luma)
                   * (1.0 - smoothstep(uLumaHigh, uLumaHigh + s, luma));
        if (uLumaInvert >= 0.5) keep = 1.0 - keep;
        alpha *= keep;
    }
    if ((uEffectMask & EFF_EXPOSURE) != 0 && uExposure != 0.0) rgb *= pow(2.0, uExposure);
    if ((uEffectMask & EFF_BRIGHTNESS) != 0 && uBrightness != 0.0) rgb += vec3(uBrightness);
    if ((uEffectMask & EFF_CONTRAST) != 0 && uContrast != 1.0) rgb = (rgb - 0.5) * uContrast + 0.5;
    if ((uEffectMask & EFF_GAMMA) != 0 && uGamma != 1.0) rgb = pow(max(rgb, vec3(0.0)), vec3(1.0 / uGamma));
    if ((uEffectMask & EFF_SATURATION) != 0 && uSaturation != 1.0) { float luma = luminance(rgb); rgb = mix(vec3(luma), rgb, uSaturation); }
    if ((uEffectMask & EFF_HUE) != 0 && uHueShift != 0.0) { vec3 hsl = rgb2hsl(rgb); hsl.x = mod(hsl.x + uHueShift / 360.0, 1.0); rgb = hsl2rgb(hsl); }
    // ---- Lift/gamma/gain wheels (Arbit color suite): applied after the
    // basic adjustments, before the grading looks.
    if ((uEffectMask & EFF_WHEELS) != 0) {
        vec3 lift = vec3(uLiftR, uLiftG, uLiftB);
        vec3 gain = vec3(uGainR, uGainG, uGainB);
        vec3 gam = max(vec3(uGammaR, uGammaG, uGammaB), vec3(1e-3));
        rgb = clamp(rgb * gain + lift * (1.0 - rgb), 0.0, 1.0);
        rgb = pow(rgb, 1.0 / gam);
    }
    if ((uEffectMask & EFF_WARM) != 0 && uWarmth > 0.0) { vec3 warm = vec3(1.1, 0.9, 0.7); rgb = mix(rgb, rgb * warm, uWarmth * 0.5); }
    if ((uEffectMask & EFF_COOL) != 0 && uCoolness > 0.0) { vec3 cool = vec3(0.8, 0.9, 1.2); rgb = mix(rgb, rgb * cool, uCoolness * 0.5); }
    if ((uEffectMask & EFF_VINTAGE) != 0 && uVintageIntensity > 0.0) { float luma = luminance(rgb); vec3 vintage = mix(vec3(luma), rgb, 0.7); vintage *= vec3(1.1, 1.0, 0.9); vintage = max(vintage, vec3(0.03)); rgb = mix(rgb, vintage, uVintageIntensity); }
    if ((uEffectMask & EFF_SEPIA) != 0 && uSepiaIntensity > 0.0) { vec3 sepia; sepia.r = dot(rgb, vec3(0.393, 0.769, 0.189)); sepia.g = dot(rgb, vec3(0.349, 0.686, 0.168)); sepia.b = dot(rgb, vec3(0.272, 0.534, 0.131)); rgb = mix(rgb, sepia, uSepiaIntensity); }
    if ((uEffectMask & EFF_BW) != 0 && uBWIntensity > 0.0) { float luma = luminance(rgb); rgb = mix(rgb, vec3(luma), uBWIntensity); }
    if ((uEffectMask & EFF_INVERT) != 0 && uInvertAmount > 0.0) rgb = mix(rgb, 1.0 - rgb, uInvertAmount);
    if ((uEffectMask & EFF_POSTERIZE) != 0 && uPosterizeLevels > 0.0) rgb = floor(rgb * uPosterizeLevels) / uPosterizeLevels;
    if ((uEffectMask & EFF_NOISE) != 0 && uNoiseAmount > 0.0) { float noise = random(TexCoord + vec2(uTime)) * 2.0 - 1.0; rgb += vec3(noise * uNoiseAmount * 0.2); }
    if ((uEffectMask & EFF_VIGNETTE) != 0 && uVignetteAmount > 0.0) { vec2 center = TexCoord - vec2(0.5); float dist = length(center) * 1.4142; float vignette = 1.0 - smoothstep(1.0 - uVignetteAmount - uVignetteSoftness, 1.0 - uVignetteAmount + 0.01, dist); rgb *= vignette; }
    rgb = clamp(rgb, 0.0, 1.0);
    FragColor = vec4(rgb, alpha);
}
)GLSL";

// Separable gaussian blur — Arbit extension. The reference editor registers
// EffectType.BLUR (bit 64, radius px) but has no sampling code anywhere
// (effects-spec §5), so this is new behavior, not parity: radius in output
// pixels, sigma = radius/2, two passes (uTexelStep = direction/outputSize).
const char* kBlurFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uTexelStep;   // one sample step (direction / output size)
uniform float uRadius;     // pixels, clamped to 20 (table max)
void main() {
    int r = int(min(uRadius, 20.0) + 0.5);
    if (r <= 0) { FragColor = texture(uTexture, TexCoord); return; }
    float sigma = max(uRadius * 0.5, 0.5);
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int i = -20; i <= 20; ++i) {
        if (i < -r || i > r) continue;
        float w = exp(-float(i * i) / (2.0 * sigma * sigma));
        sum += texture(uTexture, TexCoord + uTexelStep * float(i)) * w;
        wsum += w;
    }
    FragColor = sum / wsum;
}
)GLSL";

// Unsharp-mask sharpen — Arbit extension (reference registers
// EffectType.SHARPEN bit 128 with no implementation, effects-spec §5).
// rgb + amount * (rgb - blur3x3), clamped.
const char* kSharpenFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uTexelSize;
uniform float uAmount;
void main() {
    vec4 c = texture(uTexture, TexCoord);
    vec3 blurred = vec3(0.0);
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            blurred += texture(uTexture, TexCoord + vec2(float(dx), float(dy)) * uTexelSize).rgb;
    blurred /= 9.0;
    vec3 rgb = c.rgb + uAmount * (c.rgb - blurred);
    FragColor = vec4(clamp(rgb, 0.0, 1.0), c.a);
}
)GLSL";

// Phase 5 common production filters. One bounded native program realizes the
// exact append-only EffectType payloads as ordered ping-pong passes. It is not
// an ISF/catalog bridge and has no CPU production path.
const char* kProductionFilterFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform int uMode;
uniform vec3 uValues;
const float PI = 3.14159265358979323846;
vec4 sampleImage(vec2 uv) { return texture(uTexture, clamp(uv, 0.0, 1.0)); }
vec3 discBlur(vec2 uv, float radius, float threshold) {
    vec2 texel = 1.0 / max(uResolution, vec2(1.0));
    vec3 sum = vec3(0.0);
    float weightSum = 0.0;
    for (int i = 0; i < 24; ++i) {
        float fi = float(i) + 0.5;
        float angle = fi * 2.399963229728653;
        float distancePx = radius * sqrt(fi / 24.0);
        vec3 value = sampleImage(uv + vec2(cos(angle), sin(angle))
                                 * texel * distancePx).rgb;
        float luma = dot(value, vec3(0.2126, 0.7152, 0.0722));
        float bright = threshold <= 0.0 ? 1.0
            : smoothstep(threshold, threshold + max(threshold * 0.5, 0.05), luma);
        float weight = exp(-2.0 * distancePx * distancePx
                           / max(radius * radius, 0.25));
        sum += value * bright * weight;
        weightSum += weight;
    }
    return sum / max(weightSum, 1e-5);
}
void main() {
    vec4 center = sampleImage(TexCoord);
    if (uMode == 0) {
        vec3 glow = discBlur(TexCoord, uValues.y, 0.0);
        vec3 rgb = 1.0 - (1.0 - center.rgb) * (1.0 - clamp(glow * uValues.x, 0.0, 1.0));
        FragColor = vec4(clamp(rgb, 0.0, 1.0), center.a);
    } else if (uMode == 1) {
        vec3 glow = discBlur(TexCoord, uValues.z, uValues.x);
        FragColor = vec4(clamp(center.rgb + glow * uValues.y, 0.0, 1.0), center.a);
    } else if (uMode == 2) {
        vec3 glow = discBlur(TexCoord, uValues.y, uValues.x);
        vec3 filmRed = vec3(glow.r + glow.g * 0.45, glow.g * 0.22, glow.b * 0.04);
        FragColor = vec4(clamp(center.rgb + filmRed * uValues.z, 0.0, 1.0), center.a);
    } else if (uMode == 3) {
        vec2 aspect = vec2(uResolution.x / max(uResolution.y, 1.0), 1.0);
        vec2 p = (TexCoord - 0.5) * aspect;
        float r2 = dot(p, p);
        vec2 warped = p * (1.0 + uValues.x * r2) / aspect + 0.5;
        vec2 shift = normalize(p + vec2(1e-6)) * uValues.y * r2 / aspect;
        float red = sampleImage(warped + shift).r;
        vec4 green = sampleImage(warped);
        float blue = sampleImage(warped - shift).b;
        FragColor = vec4(red, green.g, blue, green.a);
    } else if (uMode == 4) {
        if (uValues.x <= 0.0) { FragColor = center; return; }
        int radius = int(clamp(floor(uValues.y + 0.5), 1.0, 4.0));
        vec2 texel = 1.0 / max(uResolution, vec2(1.0));
        float colorSigma = mix(0.02, 0.30, uValues.x);
        vec4 sum = vec4(0.0);
        float weightSum = 0.0;
        for (int y = -4; y <= 4; ++y)
            for (int x = -4; x <= 4; ++x) {
                if (abs(x) > radius || abs(y) > radius) continue;
                vec4 value = sampleImage(TexCoord + vec2(float(x), float(y)) * texel);
                vec3 delta = value.rgb - center.rgb;
                float spatial = exp(-float(x*x + y*y) / max(float(radius*radius), 1.0));
                float range = exp(-dot(delta, delta) / max(colorSigma * colorSigma, 1e-5));
                float weight = spatial * range;
                sum += value * weight;
                weightSum += weight;
            }
        FragColor = mix(center, sum / max(weightSum, 1e-5), uValues.x);
    } else if (uMode == 5) {
        float angle = uValues.y * PI / 180.0;
        vec2 stepUv = vec2(cos(angle), sin(angle)) / max(uResolution, vec2(1.0));
        int radius = int(min(uValues.x, 20.0) + 0.5);
        vec4 sum = vec4(0.0);
        float weightSum = 0.0;
        float sigma = max(uValues.x * 0.5, 0.5);
        for (int i = -20; i <= 20; ++i) {
            if (abs(i) > radius) continue;
            float weight = exp(-float(i*i) / (2.0 * sigma * sigma));
            sum += sampleImage(TexCoord + stepUv * float(i)) * weight;
            weightSum += weight;
        }
        FragColor = radius == 0 ? center : sum / max(weightSum, 1e-5);
    } else {
        vec3 blurred = discBlur(TexCoord, uValues.y, 0.0);
        vec3 high = center.rgb - blurred;
        float gate = uMode == 6
            ? step(uValues.z, dot(abs(high), vec3(0.2126, 0.7152, 0.0722))) : 1.0;
        FragColor = vec4(clamp(center.rgb + high * uValues.x * gate, 0.0, 1.0), center.a);
    }
}
)GLSL";

// 3D LUT pass — Arbit extension (Jun 2026 color suite). Applied after the
// effects chain (color pass + blur/sharpen), before the geometry pass.
// uLutSize = grid size N; input is remapped to texel centres
// (c * (N-1)/N + 0.5/N) so an identity .cube is bit-stable, and GL_LINEAR
// filtering on the 3D texture provides hardware trilinear interpolation.
const char* kLutFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform sampler3D uLut;
uniform float uLutSize;
void main() {
    vec4 c = texture(uTexture, TexCoord);
    vec3 uvw = clamp(c.rgb, 0.0, 1.0) * ((uLutSize - 1.0) / uLutSize) + 0.5 / uLutSize;
    FragColor = vec4(texture(uLut, uvw).rgb, c.a);
}
)GLSL";

// Window present blit: composite space keeps row 0 = image top, the window
// shows row 0 at the bottom, so the present pass flips V.
const char* kBlitVertex = R"GLSL(#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec2 aTexCoord;
out vec2 TexCoord;
void main() {
    gl_Position = vec4(aPos, 0.0, 1.0);
    TexCoord = vec2(aTexCoord.x, 1.0 - aTexCoord.y);
}
)GLSL";

const char* kBlitFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
void main() { FragColor = texture(uTexture, TexCoord); }
)GLSL";

const char* kNodePreviewFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uFinal;
uniform sampler2D uPreview;
uniform int uLayout;
uniform int uBackground;
uniform float uZoom;
uniform vec2 uPan;
uniform float uSplit;
vec3 bg(vec2 uv) {
    if (uBackground == 1) return vec3(0.0);
    if (uBackground == 2) return vec3(1.0);
    float c = mod(floor(uv.x * 32.0) + floor(uv.y * 32.0), 2.0);
    return mix(vec3(0.18), vec3(0.32), c);
}
void main() {
    if (uLayout == 1 && TexCoord.x < uSplit) {
        FragColor = texture(uFinal, vec2(TexCoord.x / uSplit, TexCoord.y)); return;
    }
    vec4 overlayRect = vec4(0.58, 0.04, 0.98, 0.44);
    if (uLayout == 2 && (TexCoord.x < overlayRect.x || TexCoord.x > overlayRect.z
                      || TexCoord.y < overlayRect.y || TexCoord.y > overlayRect.w)) {
        FragColor = texture(uFinal, TexCoord); return;
    }
    vec2 r = uLayout == 1 ? vec2((TexCoord.x-uSplit)/(1.0-uSplit), TexCoord.y)
                          : (TexCoord-overlayRect.xy)/(overlayRect.zw-overlayRect.xy);
    vec2 uv = (r-vec2(0.5))/uZoom + vec2(0.5)-uPan;
    vec4 p = (uv.x>=0.0 && uv.x<=1.0 && uv.y>=0.0 && uv.y<=1.0)
        ? texture(uPreview, uv) : vec4(0.0);
    FragColor = vec4(mix(bg(r), p.rgb, p.a), 1.0);
}
)GLSL";

// Canvas frame pass — Arbit extension (PROTOCOL.md §Project canvas & view
// transform). Post-composite, viewport-only: dims the overscan area outside
// the export frame and draws a crisp ~1.5 px accent border at the canvas
// bounds. uCanvasRect = (minU, minV, maxU, maxV) of the canvas in composite
// texture coordinates; uTexel = 1 / output size, so the border thickness is
// in output pixels regardless of the view zoom.
const char* kCanvasFrameFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec4 uCanvasRect;
uniform vec2 uTexel;
void main() {
    vec4 c = texture(uTexture, TexCoord);
    bool inside = TexCoord.x >= uCanvasRect.x && TexCoord.x <= uCanvasRect.z
               && TexCoord.y >= uCanvasRect.y && TexCoord.y <= uCanvasRect.w;
    if (!inside)
        c.rgb *= 0.45;   // dim everything outside the export frame
    const float bw = 1.5; // border half-thickness in output pixels
    float dx = min(abs(TexCoord.x - uCanvasRect.x), abs(TexCoord.x - uCanvasRect.z)) / uTexel.x;
    float dy = min(abs(TexCoord.y - uCanvasRect.y), abs(TexCoord.y - uCanvasRect.w)) / uTexel.y;
    bool spanX = TexCoord.x >= uCanvasRect.x - bw * uTexel.x && TexCoord.x <= uCanvasRect.z + bw * uTexel.x;
    bool spanY = TexCoord.y >= uCanvasRect.y - bw * uTexel.y && TexCoord.y <= uCanvasRect.w + bw * uTexel.y;
    if ((dx <= bw && spanY) || (dy <= bw && spanX))
        c = vec4(0.365, 0.663, 1.0, 1.0); // canvas blue; layer selection stays violet
    FragColor = c;
}
)GLSL";

// HDR bloom bright-pass (visuals-quality increment 2): extracts the part of the
// HDR composite above a luminance threshold, with a quadratic soft knee so the
// bloom fades in smoothly rather than hard-clipping at the threshold. Output
// keeps HDR magnitude (the target is RGBA16F) so the blur and combine operate
// in linear light. Colour is preserved (the knee scales the original rgb).
const char* kBloomThreshFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float uThreshold;
void main() {
    vec3 c = texture(uTexture, TexCoord).rgb;
    float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
    float knee = max(uThreshold * 0.5, 1e-4);
    float k = clamp((lum - uThreshold + knee) / (2.0 * knee), 0.0, 1.0);
    FragColor = vec4(c * (k * k), 1.0);
}
)GLSL";

// HDR combine + tonemap (increment 2): final = tonemap((composite + bloom *
// intensity) * exposure). tonemap 0 = hard clamp (linear; identical to the
// old readback behaviour for in-range content), 1 = Reinhard, 2 = ACES filmic
// (Narkowicz approximation). Writes opaque LDR ready for the 8-bit readback.
const char* kBloomCombineFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;     // HDR composite
uniform sampler2D uBloomTex;    // blurred bright-pass (HDR)
uniform float uIntensity;
uniform float uExposure;
uniform int uTonemap;
vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}
void main() {
    vec3 base  = texture(uTexture, TexCoord).rgb;
    vec3 bloom = texture(uBloomTex, TexCoord).rgb;
    vec3 hdr = (base + bloom * uIntensity) * uExposure;
    vec3 ldr;
    if (uTonemap == 1)      ldr = hdr / (hdr + vec3(1.0));
    else if (uTonemap == 2) ldr = aces(hdr);
    else                    ldr = clamp(hdr, 0.0, 1.0);
    FragColor = vec4(ldr, 1.0);
}
)GLSL";

// The shader's EFF_* constants are hardcoded verbatim above; make sure they
// stay in lockstep with the canonical table.
static_assert (videofx::kEffectBits[(int) videofx::EffectType::Brightness] == 1
            && videofx::kEffectBits[(int) videofx::EffectType::Contrast] == 2
            && videofx::kEffectBits[(int) videofx::EffectType::Saturation] == 4
            && videofx::kEffectBits[(int) videofx::EffectType::Hue] == 8
            && videofx::kEffectBits[(int) videofx::EffectType::Exposure] == 16
            && videofx::kEffectBits[(int) videofx::EffectType::Gamma] == 32
            && videofx::kEffectBits[(int) videofx::EffectType::Blur] == 64
            && videofx::kEffectBits[(int) videofx::EffectType::Sharpen] == 128
            && videofx::kEffectBits[(int) videofx::EffectType::Vignette] == 256
            && videofx::kEffectBits[(int) videofx::EffectType::Warm] == 512
            && videofx::kEffectBits[(int) videofx::EffectType::Cool] == 1024
            && videofx::kEffectBits[(int) videofx::EffectType::Vintage] == 2048
            && videofx::kEffectBits[(int) videofx::EffectType::Sepia] == 4096
            && videofx::kEffectBits[(int) videofx::EffectType::BlackAndWhite] == 8192
            && videofx::kEffectBits[(int) videofx::EffectType::Invert] == 16384
            && videofx::kEffectBits[(int) videofx::EffectType::Posterize] == 32768
            && videofx::kEffectBits[(int) videofx::EffectType::Noise] == 65536
            && videofx::kEffectBits[(int) videofx::EffectType::ChromaKey] == 131072
            && videofx::kEffectBits[(int) videofx::EffectType::LumaKey] == 262144
            && videofx::kEffectBits[(int) videofx::EffectType::ColorWheels] == 524288,
            "videofx::kEffectBits must match the uber-shader EFF_* constants");

// Reference quad geometry — verbatim (compositor-spec §2): interleaved
// pos.xy + uv.xy, v=0 on the bottom-left vertex.
const float kQuadVertices[16] = {
    -1.0f, -1.0f, 0.0f, 0.0f,  // Bottom-left
     1.0f, -1.0f, 1.0f, 0.0f,  // Bottom-right
     1.0f,  1.0f, 1.0f, 1.0f,  // Top-right
    -1.0f,  1.0f, 0.0f, 1.0f,  // Top-left
};
const unsigned kQuadIndices[6] = { 0, 1, 2, 0, 2, 3 };

const float kIdentity[16] = {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1,
};

// M = T(tx,ty) · R(deg) · S(sx,sy), column-major (compositor-spec §3.2).
void buildTransform (float tx, float ty, float rotationDeg,
                     float sx, float sy, float out[16])
{
    const float a = rotationDeg * 3.14159265358979323846f / 180.0f;
    const float c = std::cos (a);
    const float s = std::sin (a);
    const float m[16] = {
        sx * c,  sx * s,  0.0f, 0.0f,
       -sy * s,  sy * c,  0.0f, 0.0f,
        0.0f,    0.0f,    1.0f, 0.0f,
        tx,      ty,      0.0f, 1.0f,
    };
    std::memcpy (out, m, sizeof (m));
}

// Wire transition type (videofx::TransitionType: 0 none, 1 fade, 2 dissolve,
// 3..6 wipes l/r/u/d) → reference shader index (0 dissolve, 1 fade,
// 2..5 wipes). Unknown types fall back to dissolve like the reference's
// type_map.get(t, 0). Returns -1 for "no transition".
int transitionShaderType (int wireType)
{
    switch (wireType)
    {
        case 0:  return -1;
        case 1:  return 1;  // fade
        case 2:  return 0;  // dissolve
        case 3:  return 2;  // wipe left
        case 4:  return 3;  // wipe right
        case 5:  return 4;  // wipe up
        case 6:  return 5;  // wipe down
        default: return 0;  // unknown -> dissolve (reference fallback)
    }
}

// Uniform index order for the effect value uniforms (upload order from
// effects-spec §8 step 7/8 for the original 16; color-suite uniforms appended).
enum FxUniform
{
    FxBrightness = 0, FxContrast, FxSaturation, FxHueShift, FxExposure,
    FxGamma, FxVignetteAmount, FxVignetteSoftness, FxWarmth, FxCoolness,
    FxVintage, FxSepia, FxBW, FxInvert, FxPosterize, FxNoise,
    // Color suite (Arbit extension)
    FxKeyR, FxKeyG, FxKeyB, FxKeyTolerance, FxKeySoftness, FxKeySpill,
    FxLumaLow, FxLumaHigh, FxLumaSoftness, FxLumaInvert,
    FxLiftR, FxLiftG, FxLiftB, FxGammaR, FxGammaG, FxGammaB,
    FxGainR, FxGainG, FxGainB,
    FxKeyChoke, FxKeyFeather, FxKeyEdgeR, FxKeyEdgeG, FxKeyEdgeB,
    FxKeyEdgeAmount, FxKeyMatteView, FxKeyTexelX, FxKeyTexelY,
    FxUniformCount
};

const char* kFxUniformNames[FxUniformCount] = {
    "uBrightness", "uContrast", "uSaturation", "uHueShift", "uExposure",
    "uGamma", "uVignetteAmount", "uVignetteSoftness", "uWarmth", "uCoolness",
    "uVintageIntensity", "uSepiaIntensity", "uBWIntensity", "uInvertAmount",
    "uPosterizeLevels", "uNoiseAmount",
    "uKeyR", "uKeyG", "uKeyB", "uKeyTolerance", "uKeySoftness", "uKeySpill",
    "uLumaLow", "uLumaHigh", "uLumaSoftness", "uLumaInvert",
    "uLiftR", "uLiftG", "uLiftB", "uGammaR", "uGammaG", "uGammaB",
    "uGainR", "uGainG", "uGainB",
    "uKeyChoke", "uKeyFeather", "uKeyEdgeR", "uKeyEdgeG", "uKeyEdgeB",
    "uKeyEdgeAmount", "uKeyMatteView", "uKeyTexelX", "uKeyTexelY",
};

// Neutral defaults (effects-spec §8 step 7): uploaded for every uniform a
// present effect does not override. Note posterize stays 8.0 when absent.
const float kFxNeutral[FxUniformCount] = {
    0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.5f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 8.0f, 0.0f,
    // key colour/tolerance/softness/spill
    0.0f, 1.0f, 0.0f, 0.18f, 0.10f, 0.0f,
    // luma low/high/softness/invert (keep everything)
    0.0f, 1.0f, 0.1f, 0.0f,
    // wheels lift/gamma/gain
    0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    // key cleanup choke/feather/edge/matte-view/texel
    0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f,
};

static_assert (FxUniformCount <= 48, "FxProg::uValue too small");

} // namespace

// ---------------------------------------------------------------------------

unsigned FrameRenderer::compileProgram (const char* vertexSrc, const char* fragmentSrc,
                                        const char* name, std::string& error) const
{
    auto compileStage = [&] (GLenum type, const char* src) -> unsigned
    {
        const unsigned sh = gl_->CreateShader (type);
        gl_->ShaderSource (sh, 1, &src, nullptr);
        gl_->CompileShader (sh);
        GLint ok = GL_FALSE;
        gl_->GetShaderiv (sh, GL_COMPILE_STATUS, &ok);
        if (ok != GL_TRUE)
        {
            char log[2048] = {};
            GLsizei len = 0;
            gl_->GetShaderInfoLog (sh, sizeof (log) - 1, &len, log);
            error = std::string (name) + " "
                  + (type == GL_VERTEX_SHADER ? "vertex" : "fragment")
                  + " shader compile failed: " + log;
            gl_->DeleteShader (sh);
            return 0;
        }
        return sh;
    };

    const unsigned vs = compileStage (GL_VERTEX_SHADER, vertexSrc);
    if (vs == 0) return 0;
    const unsigned fs = compileStage (GL_FRAGMENT_SHADER, fragmentSrc);
    if (fs == 0) { gl_->DeleteShader (vs); return 0; }

    const unsigned prog = gl_->CreateProgram();
    gl_->AttachShader (prog, vs);
    gl_->AttachShader (prog, fs);
    gl_->LinkProgram (prog);
    gl_->DeleteShader (vs);
    gl_->DeleteShader (fs);

    GLint ok = GL_FALSE;
    gl_->GetProgramiv (prog, GL_LINK_STATUS, &ok);
    if (ok != GL_TRUE)
    {
        char log[2048] = {};
        GLsizei len = 0;
        gl_->GetProgramInfoLog (prog, sizeof (log) - 1, &len, log);
        error = std::string (name) + " program link failed: " + log;
        gl_->DeleteProgram (prog);
        return 0;
    }
    return prog;
}

// ---- Geometric / UV effect-pass fragment shaders (Jun 2026, Arbit extension).
// Run as per-slot ping-pong passes AFTER the color/blur/sharpen/LUT chain (see
// runEffectChain), each reading the previous result via uTexture/TexCoord.
// Deliberately standalone passes, NOT branches in the color uber-shader (the
// geometric family stays per-pass — master-plan principle). Each shader's float
// uniform names equal the videofx::kEffectDefs param wire names, so the init
// loop binds locations straight from that table. They share kVertexShader, so
// uTransform/uCrop are uploaded by setPassthrough exactly like blur/sharpen.

// Fold UV into N mirrored wedges around the (aspect-corrected) centre; angle is
// a static rotation in turns, spin animates it against deterministic uTime.
const char* kKaleidoscopeFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float uTime;
uniform float segments;
uniform float angle;
uniform float spin;
uniform float zoom;
void main() {
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 p = (TexCoord - 0.5) * vec2(aspect, 1.0);
    float r = length(p);
    float a = atan(p.y, p.x);
    float wedge = 6.28318530718 / max(segments, 2.0);
    a -= (angle + spin * uTime) * 6.28318530718;
    a = mod(a, wedge);
    a = abs(a - 0.5 * wedge);
    vec2 q = vec2(cos(a), sin(a)) * (r / max(zoom, 0.01));
    vec2 uv = q / vec2(aspect, 1.0) + 0.5;
    FragColor = texture(uTexture, clamp(uv, 0.0, 1.0));
}
)GLSL";

// Reflect across the centre line(s): 0 left->right, 1 right->left, 2 top->bottom,
// 3 quad (both axes).
const char* kMirrorFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float mode;
void main() {
    vec2 uv = TexCoord;
    int m = int(mode + 0.5);
    if (m == 0)      { if (uv.x > 0.5) uv.x = 1.0 - uv.x; }
    else if (m == 1) { if (uv.x < 0.5) uv.x = 1.0 - uv.x; }
    else if (m == 2) { if (uv.y > 0.5) uv.y = 1.0 - uv.y; }
    else             { if (uv.x > 0.5) uv.x = 1.0 - uv.x;
                       if (uv.y > 0.5) uv.y = 1.0 - uv.y; }
    FragColor = texture(uTexture, uv);
}
)GLSL";

// Repeat the source in an N x N grid; mirrorEdges flips alternate cells so the
// tiling is seamless.
const char* kTileFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float count;
uniform float mirrorEdges;
void main() {
    float n = max(count, 1.0);
    vec2 uv = TexCoord * n;
    vec2 cell = floor(uv);
    uv = fract(uv);
    if (mirrorEdges > 0.5) {
        if (mod(cell.x, 2.0) >= 1.0) uv.x = 1.0 - uv.x;
        if (mod(cell.y, 2.0) >= 1.0) uv.y = 1.0 - uv.y;
    }
    FragColor = texture(uTexture, uv);
}
)GLSL";

// Sinusoidal UV wobble; speed animates it against deterministic uTime.
const char* kWarpFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float uTime;
uniform float amount;
uniform float frequency;
uniform float speed;
void main() {
    vec2 uv = TexCoord;
    float t = uTime * speed;
    uv.x += amount * sin(uv.y * frequency + t);
    uv.y += amount * cos(uv.x * frequency + t * 1.3);
    FragColor = texture(uTexture, clamp(uv, 0.0, 1.0));
}
)GLSL";

// Push each pixel along the source's luminance gradient (relief / emboss feel).
const char* kDisplaceFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float amount;
void main() {
    vec2 texel = 1.0 / max(uResolution, vec2(1.0));
    vec3 lw = vec3(0.299, 0.587, 0.114);
    float lx = dot(texture(uTexture, TexCoord + vec2(texel.x, 0.0)).rgb, lw)
             - dot(texture(uTexture, TexCoord - vec2(texel.x, 0.0)).rgb, lw);
    float ly = dot(texture(uTexture, TexCoord + vec2(0.0, texel.y)).rgb, lw)
             - dot(texture(uTexture, TexCoord - vec2(0.0, texel.y)).rgb, lw);
    vec2 uv = clamp(TexCoord + vec2(lx, ly) * amount, 0.0, 1.0);
    FragColor = texture(uTexture, uv);
}
)GLSL";

// Rotate UVs about the centre, with more rotation toward the middle and none
// beyond `radius` — a localized swirl/vortex. amount is in turns at the centre.
const char* kPolarSwirlFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float amount;
uniform float radius;
void main() {
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    vec2 p = (TexCoord - 0.5) * vec2(aspect, 1.0);
    float r = length(p);
    float falloff = 1.0 - smoothstep(0.0, max(radius, 1e-3), r);
    float a = amount * 6.28318530718 * falloff;
    float c = cos(a), s = sin(a);
    vec2 q = mat2(c, -s, s, c) * p;
    vec2 uv = q / vec2(aspect, 1.0) + 0.5;
    FragColor = texture(uTexture, clamp(uv, 0.0, 1.0));
}
)GLSL";

// Chromatic shift: sample R and B from opposite offsets along `angle` (turns),
// leaving G centred — a directional RGB split / chromatic-aberration look.
const char* kDisplaceRgbFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform float amount;
uniform float angle;
void main() {
    float a = angle * 6.28318530718;
    vec2 dir = vec2(cos(a), sin(a)) * amount;
    float r = texture(uTexture, clamp(TexCoord + dir, 0.0, 1.0)).r;
    vec4 g = texture(uTexture, TexCoord);
    float b = texture(uTexture, clamp(TexCoord - dir, 0.0, 1.0)).b;
    FragColor = vec4(r, g.g, b, g.a);
}
)GLSL";

// Snap UVs to a coarse grid (block centre sampled) — mosaic / pixelation.
// `size` is the block edge in source pixels.
const char* kPixelateFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform vec2 uResolution;
uniform float size;
void main() {
    float blk = max(size, 1.0);
    vec2 res = max(uResolution, vec2(1.0));
    vec2 uv = (floor(TexCoord * res / blk) + 0.5) * blk / res;
    FragColor = texture(uTexture, clamp(uv, 0.0, 1.0));
}
)GLSL";

// Feedback trail — the one CROSS-FRAME pass. uTexture is the current
// (pre-feedback) frame; uFeedbackTex is THIS clip's previous-frame output from
// a persistent history FBO (PINGPONG.md §2). Screen-blends the current frame
// over a decayed, zoomed + swirled copy of the trail, so motion leaves a
// luminous wake. decay = trail length; zoom/swirl = how the trail drifts each
// frame. Determinism: zoom/swirl are fixed per rendered frame, so matched-fps
// linear playback and export produce identical trails (the documented parity
// exception covers scrub/seek).
const char* kFeedbackFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform sampler2D uFeedbackTex;
uniform float decay;
uniform float zoom;
uniform float swirl;
void main() {
    vec2 p = TexCoord - 0.5;
    float c = cos(swirl), s = sin(swirl);
    mat2 R = mat2(c, -s, s, c);
    vec2 tuv = (R * p) * zoom + 0.5;
    vec4 prev = texture(uFeedbackTex, clamp(tuv, 0.0, 1.0)) * clamp(decay, 0.0, 0.999);
    vec4 cur  = texture(uTexture, TexCoord);
    FragColor = 1.0 - (1.0 - cur) * (1.0 - prev);
    FragColor.a = max(cur.a, prev.a);
}
)GLSL";

// Frame Blend fragment shader (retime tier 1): straight alpha-mix of the two
// bracket frames at TexCoord. mix 0 = texA (earlier frame), 1 = texB (later).
// Identity transform/zero crop (setPassthrough) ⇒ mix=0 is a pixel-identical
// copy of texA, so a frame-blended layer is orientation-compatible with the
// nearest-frame upload the compositor otherwise consumes.
const char* kFrameBlendFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexA;
uniform sampler2D uTexB;
uniform float uMix;
void main() {
    FragColor = mix(texture(uTexA, TexCoord), texture(uTexB, TexCoord),
                    clamp(uMix, 0.0, 1.0));
}
)GLSL";

// Velocity-buffer motion blur. Motion is RG16F pixel displacement in image
// coordinates. The fixed loop bound keeps GLSL 330 drivers predictable while
// uSampleCount selects the exact admitted count.
const char* kMotionBlurFragment = R"GLSL(#version 330 core
in vec2 TexCoord;
out vec4 FragColor;
uniform sampler2D uTexture;
uniform sampler2D uMotion;
uniform vec2 uResolution;
uniform int uSampleCount;
uniform float uShutterScale;
void main()
{
    vec2 velocityUv = texture(uMotion, TexCoord).rg
                    / max(uResolution, vec2(1.0));
    velocityUv *= uShutterScale;
    vec4 sum = vec4(0.0);
    int count = max(uSampleCount, 1);
    for (int i = 0; i < 64; ++i)
    {
        if (i >= count) break;
        float position = count == 1 ? 0.0
            : float(i) / float(count - 1) - 0.5;
        sum += texture(uTexture, clamp(TexCoord + velocityUv * position,
                                      vec2(0.0), vec2(1.0)));
    }
    FragColor = sum / float(count);
}
)GLSL";

namespace
{
// Stateless single-pass geometric effects. Each maps an EffectType to its
// fragment shader; geom_[] is indexed by position in THIS table (dense), not
// by enum value — so the effects need not be contiguous in EffectType (the
// cross-frame FeedbackTrail sits between this block and later additions, and
// effect_defs.h is append-only). All share the uTexture/uResolution/uTime
// uniform set plus their own params; absent uniforms resolve to -1 (ignored).
struct GeomEffectEntry { videofx::EffectType type; const char* frag; const char* name; };
const GeomEffectEntry kGeomEffects[] = {
    { videofx::EffectType::Kaleidoscope, kKaleidoscopeFragment, "kaleidoscope" },
    { videofx::EffectType::Mirror,       kMirrorFragment,       "mirror" },
    { videofx::EffectType::Tile,         kTileFragment,         "tile" },
    { videofx::EffectType::Warp,         kWarpFragment,         "warp" },
    { videofx::EffectType::Displace,     kDisplaceFragment,     "displace" },
    { videofx::EffectType::PolarSwirl,   kPolarSwirlFragment,   "polar_swirl" },
    { videofx::EffectType::DisplaceRgb,  kDisplaceRgbFragment,  "displace_rgb" },
    { videofx::EffectType::Pixelate,     kPixelateFragment,     "pixelate" },
};
// Dense geom_[] index for an EffectType, or -1 if it is not a stateless geom
// effect (color/blur/feedback/etc. are handled elsewhere).
inline int geomIndexForType (int type)
{
    for (int i = 0; i < (int) (sizeof (kGeomEffects) / sizeof (kGeomEffects[0])); ++i)
        if ((int) kGeomEffects[i].type == type) return i;
    return -1;
}

} // namespace

bool FrameRenderer::initializeForVisualPlans (
    const arbitgl::GlFuncs* gl, int outWidth, int outHeight,
    const std::vector<videowire::CompiledVisualLayerPlan>& plans,
    std::string& error, bool metalOnly,
    int clipId, unsigned sourceTexture, LayerDesc* output)
{
    if (output != nullptr)
        *output = {};
    error.clear();

    videowire::VisualPlanExecutionState state;
    if (! state.admitPlans(plans, &error, outWidth, outHeight))
        return false;

    if (output != nullptr)
    {
        LayerDesc candidate;
        candidate.texture = sourceTexture;
        candidate.texWidth = outWidth;
        candidate.texHeight = outHeight;
        if (! videowire::executeVisualLayerPlan(
                plans, clipId, candidate, error, videohelper::geometry::PlanUse::preview, nullptr, nullptr, &state))
            return false;
        *output = candidate;
    }

    return initialize(gl, outWidth, outHeight, error, metalOnly);
}

bool FrameRenderer::initialize (const arbitgl::GlFuncs* gl, int outWidth, int outHeight,
                                std::string& error, bool metalOnly)
{
    static_assert (sizeof (kGeomEffects) / sizeof (kGeomEffects[0]) == kGeomEffectCount,
                   "kGeomEffects table size must equal kGeomEffectCount (sizes geom_[])");
    gl_ = gl;
    metalOnly_ = metalOnly;
    if (gl_ == nullptr && ! metalOnly_)
    {
        error = "null GL function table";
        return false;
    }

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        outW_ = std::max (outWidth, 1);
        outH_ = std::max (outHeight, 1);
        metalRenderer_ = std::make_unique<MetalFrameRenderer>();
        if (! metalRenderer_->initialize (gl_, outW_, outH_, error, true))
        {
            metalRenderer_.reset();
            gl_ = nullptr;
            return false;
        }
        error.clear();
        return true;
    }
#else
    if (metalOnly_)
    {
        error = "Metal-only renderer requested without Metal support";
        gl_ = nullptr;
        return false;
    }
#endif

    if ((layer_.prog = compileProgram (kVertexShader, kLayerFragment, "layer", error)) == 0
        || (drawShape_.prog = compileProgram (kVertexShader, kDrawShapeFragment, "drawShape", error)) == 0
        || (blend_.prog = compileProgram (kVertexShader, kBlendFragment, "blend", error)) == 0
        || (trans_.prog = compileProgram (kVertexShader, kTransitionFragment, "transition", error)) == 0
        || (fx_.prog = compileProgram (kVertexShader, kEffectsFragment, "effects", error)) == 0
        || (blur_.prog = compileProgram (kVertexShader, kBlurFragment, "blur", error)) == 0
        || (sharpen_.prog = compileProgram (kVertexShader, kSharpenFragment, "sharpen", error)) == 0
        || (productionFilter_.prog = compileProgram (
                kVertexShader, kProductionFilterFragment, "productionFilter", error)) == 0
        || (lut_.prog = compileProgram (kVertexShader, kLutFragment, "lut", error)) == 0
        || (blit_.prog = compileProgram (kBlitVertex, kBlitFragment, "blit", error)) == 0
        || (preview_.prog = compileProgram (kVertexShader, kNodePreviewFragment, "nodePreview", error)) == 0
        || (canvas_.prog = compileProgram (kVertexShader, kCanvasFrameFragment, "canvasFrame", error)) == 0
        || (bloomThresh_.prog = compileProgram (kVertexShader, kBloomThreshFragment, "bloomThresh", error)) == 0
        || (bloomCombine_.prog = compileProgram (kVertexShader, kBloomCombineFragment, "bloomCombine", error)) == 0)
    {
        shutdown();
        return false;
    }

    // Geometric/UV effect-pass programs (Jun 2026). A bad geometric shader must
    // not silently disable the rest of the chain, so a compile failure here is
    // a hard init failure exactly like every other program above.
    for (int g = 0; g < kGeomEffectCount; ++g)
        if ((geom_[g].prog = compileProgram (kVertexShader, kGeomEffects[g].frag, kGeomEffects[g].name, error)) == 0)
        {
            shutdown();
            return false;
        }

    if ((feedback_.prog = compileProgram (kVertexShader, kFeedbackFragment, "feedback_trail", error)) == 0)
    {
        shutdown();
        return false;
    }

    if ((frameBlend_.prog = compileProgram (kVertexShader, kFrameBlendFragment, "frameBlend", error)) == 0)
    {
        shutdown();
        return false;
    }
    if ((motionBlur_.prog = compileProgram (kVertexShader, kMotionBlurFragment, "motionBlur", error)) == 0)
    {
        shutdown();
        return false;
    }

    auto loc = [&] (unsigned prog, const char* n) { return gl_->GetUniformLocation (prog, n); };
    layer_.uTransform = loc (layer_.prog, "uTransform");
    layer_.uCrop      = loc (layer_.prog, "uCrop");
    layer_.uCornerPin = loc (layer_.prog, "uCornerPin");
    layer_.uCorners   = loc (layer_.prog, "uCorners");
    layer_.uTexture   = loc (layer_.prog, "uTexture");
    layer_.uOpacity   = loc (layer_.prog, "uOpacity");
    layer_.uBlendMode = loc (layer_.prog, "uBlendMode");
    layer_.uMaskType    = loc (layer_.prog, "uMaskType");
    layer_.uMaskRect    = loc (layer_.prog, "uMaskRect");
    layer_.uMaskFeather = loc (layer_.prog, "uMaskFeather");
    layer_.uMaskInvert  = loc (layer_.prog, "uMaskInvert");
    layer_.uPathRect = loc (layer_.prog, "uPathRect");
    layer_.uPathRectB = loc (layer_.prog, "uPathRectB");
    layer_.uPathOperation = loc (layer_.prog, "uPathOperation");
    layer_.uPathInvert = loc (layer_.prog, "uPathInvert");
    layer_.uMatteTexture = loc (layer_.prog, "uMatteTexture");
    layer_.uMatteTextureB = loc (layer_.prog, "uMatteTextureB");
    layer_.uMatteCombineMode = loc (layer_.prog, "uMatteCombineMode");
    layer_.uMatteApply = loc (layer_.prog, "uMatteApply");
    layer_.uMatteTexel = loc (layer_.prog, "uMatteTexel");
    layer_.uMatteRefine = loc (layer_.prog, "uMatteRefine");
    layer_.uMatteChoke = loc (layer_.prog, "uMatteChoke");
    layer_.uMatteInvert = loc (layer_.prog, "uMatteInvert");
    layer_.uDepthTexture = loc (layer_.prog, "uDepthTexture");
    layer_.uDepthFog = loc (layer_.prog, "uDepthFog");
    layer_.uFogRangeDensity = loc (layer_.prog, "uFogRangeDensity");
    layer_.uFogColor = loc (layer_.prog, "uFogColor");

    drawShape_.uTransform = loc (drawShape_.prog, "uTransform");
    drawShape_.uCrop      = loc (drawShape_.prog, "uCrop");
    drawShape_.uRect      = loc (drawShape_.prog, "uShapeRect");
    drawShape_.uRectB     = loc (drawShape_.prog, "uShapeRectB");
    drawShape_.uOperation = loc (drawShape_.prog, "uShapeOperation");
    drawShape_.uColor     = loc (drawShape_.prog, "uShapeColor");

    blend_.uTransform = loc (blend_.prog, "uTransform");
    blend_.uCrop      = loc (blend_.prog, "uCrop");
    blend_.uFrontTex  = loc (blend_.prog, "uFrontTex");
    blend_.uBackTex   = loc (blend_.prog, "uBackTex");
    blend_.uOpacity   = loc (blend_.prog, "uOpacity");
    blend_.uBlendMode = loc (blend_.prog, "uBlendMode");

    frameBlend_.uTransform = loc (frameBlend_.prog, "uTransform");
    frameBlend_.uCrop      = loc (frameBlend_.prog, "uCrop");
    frameBlend_.uTexA      = loc (frameBlend_.prog, "uTexA");
    frameBlend_.uTexB      = loc (frameBlend_.prog, "uTexB");
    frameBlend_.uMix       = loc (frameBlend_.prog, "uMix");

    motionBlur_.uTransform = loc (motionBlur_.prog, "uTransform");
    motionBlur_.uCrop = loc (motionBlur_.prog, "uCrop");
    motionBlur_.uTexture = loc (motionBlur_.prog, "uTexture");
    motionBlur_.uMotion = loc (motionBlur_.prog, "uMotion");
    motionBlur_.uResolution = loc (motionBlur_.prog, "uResolution");
    motionBlur_.uSampleCount = loc (motionBlur_.prog, "uSampleCount");
    motionBlur_.uShutterScale = loc (motionBlur_.prog, "uShutterScale");

    trans_.uTransform = loc (trans_.prog, "uTransform");
    trans_.uCrop      = loc (trans_.prog, "uCrop");
    trans_.uFromTex   = loc (trans_.prog, "uFromTex");
    trans_.uToTex     = loc (trans_.prog, "uToTex");
    trans_.uProgress  = loc (trans_.prog, "uProgress");
    trans_.uType      = loc (trans_.prog, "uTransitionType");

    fx_.uTransform  = loc (fx_.prog, "uTransform");
    fx_.uCrop       = loc (fx_.prog, "uCrop");
    fx_.uTexture    = loc (fx_.prog, "uTexture");
    fx_.uTime       = loc (fx_.prog, "uTime");
    fx_.uEffectMask = loc (fx_.prog, "uEffectMask");
    for (int i = 0; i < FxUniformCount; ++i)
        fx_.uValue[i] = loc (fx_.prog, kFxUniformNames[i]);

    blur_.uTransform = loc (blur_.prog, "uTransform");
    blur_.uCrop      = loc (blur_.prog, "uCrop");
    blur_.uTexture   = loc (blur_.prog, "uTexture");
    blur_.uTexelStep = loc (blur_.prog, "uTexelStep");
    blur_.uRadius    = loc (blur_.prog, "uRadius");

    sharpen_.uTransform = loc (sharpen_.prog, "uTransform");
    sharpen_.uCrop      = loc (sharpen_.prog, "uCrop");
    sharpen_.uTexture   = loc (sharpen_.prog, "uTexture");
    sharpen_.uTexelSize = loc (sharpen_.prog, "uTexelSize");
    sharpen_.uAmount    = loc (sharpen_.prog, "uAmount");

    productionFilter_.uTransform = loc (productionFilter_.prog, "uTransform");
    productionFilter_.uCrop = loc (productionFilter_.prog, "uCrop");
    productionFilter_.uTexture = loc (productionFilter_.prog, "uTexture");
    productionFilter_.uResolution = loc (productionFilter_.prog, "uResolution");
    productionFilter_.uMode = loc (productionFilter_.prog, "uMode");
    productionFilter_.uValues = loc (productionFilter_.prog, "uValues");

    lut_.uTransform = loc (lut_.prog, "uTransform");
    lut_.uCrop      = loc (lut_.prog, "uCrop");
    lut_.uTexture   = loc (lut_.prog, "uTexture");
    lut_.uLut       = loc (lut_.prog, "uLut");
    lut_.uLutSize   = loc (lut_.prog, "uLutSize");

    blit_.uTexture = loc (blit_.prog, "uTexture");
    preview_.uFinal = loc (preview_.prog, "uFinal");
    preview_.uPreview = loc (preview_.prog, "uPreview");
    preview_.uLayout = loc (preview_.prog, "uLayout");
    preview_.uBackground = loc (preview_.prog, "uBackground");
    preview_.uZoom = loc (preview_.prog, "uZoom");
    preview_.uPan = loc (preview_.prog, "uPan");
    preview_.uSplit = loc (preview_.prog, "uSplit");

    canvas_.uTransform  = loc (canvas_.prog, "uTransform");
    canvas_.uCrop       = loc (canvas_.prog, "uCrop");
    canvas_.uTexture    = loc (canvas_.prog, "uTexture");
    canvas_.uCanvasRect = loc (canvas_.prog, "uCanvasRect");
    canvas_.uTexel      = loc (canvas_.prog, "uTexel");

    bloomThresh_.uTransform = loc (bloomThresh_.prog, "uTransform");
    bloomThresh_.uCrop      = loc (bloomThresh_.prog, "uCrop");
    bloomThresh_.uTexture   = loc (bloomThresh_.prog, "uTexture");
    bloomThresh_.uThreshold = loc (bloomThresh_.prog, "uThreshold");

    bloomCombine_.uTransform = loc (bloomCombine_.prog, "uTransform");
    bloomCombine_.uCrop      = loc (bloomCombine_.prog, "uCrop");
    bloomCombine_.uTexture   = loc (bloomCombine_.prog, "uTexture");
    bloomCombine_.uBloomTex  = loc (bloomCombine_.prog, "uBloomTex");
    bloomCombine_.uIntensity = loc (bloomCombine_.prog, "uIntensity");
    bloomCombine_.uExposure  = loc (bloomCombine_.prog, "uExposure");
    bloomCombine_.uTonemap   = loc (bloomCombine_.prog, "uTonemap");

    // Geometric passes: shared transform/crop/texture + resolution/time, then
    // each param uniform fetched by its wire name straight from the table
    // (uniform name == kEffectDefs param name). Unused uniforms resolve to -1
    // and their later glUniform calls are silently ignored.
    for (int g = 0; g < kGeomEffectCount; ++g)
    {
        geom_[g].uTransform  = loc (geom_[g].prog, "uTransform");
        geom_[g].uCrop       = loc (geom_[g].prog, "uCrop");
        geom_[g].uTexture    = loc (geom_[g].prog, "uTexture");
        geom_[g].uResolution = loc (geom_[g].prog, "uResolution");
        geom_[g].uTime       = loc (geom_[g].prog, "uTime");
        const videofx::EffectDef* def = videofx::effectDefFor ((int) kGeomEffects[g].type);
        const int pc = def != nullptr ? def->paramCount : 0;
        for (int j = 0; j < pc && j < videofx::kMaxEffectParams; ++j)
            geom_[g].uParam[j] = loc (geom_[g].prog, def->params[j].name);
    }

    feedback_.uTransform   = loc (feedback_.prog, "uTransform");
    feedback_.uCrop        = loc (feedback_.prog, "uCrop");
    feedback_.uTexture     = loc (feedback_.prog, "uTexture");
    feedback_.uFeedbackTex = loc (feedback_.prog, "uFeedbackTex");
    {
        const videofx::EffectDef* def = videofx::effectDefFor ((int) videofx::EffectType::FeedbackTrail);
        const int pc = def != nullptr ? def->paramCount : 0;
        for (int j = 0; j < pc && j < videofx::kMaxEffectParams; ++j)
            feedback_.uParam[j] = loc (feedback_.prog, def->params[j].name);
    }

    // Shared fullscreen quad (one VAO/VBO/EBO for every pass).
    gl_->GenVertexArrays (1, &vao_);
    gl_->BindVertexArray (vao_);
    gl_->GenBuffers (1, &vbo_);
    gl_->BindBuffer (GL_ARRAY_BUFFER, vbo_);
    gl_->BufferData (GL_ARRAY_BUFFER, sizeof (kQuadVertices), kQuadVertices, GL_STATIC_DRAW);
    gl_->GenBuffers (1, &ebo_);
    gl_->BindBuffer (GL_ELEMENT_ARRAY_BUFFER, ebo_);
    gl_->BufferData (GL_ELEMENT_ARRAY_BUFFER, sizeof (kQuadIndices), kQuadIndices, GL_STATIC_DRAW);
    gl_->VertexAttribPointer (0, 2, GL_FLOAT, GL_FALSE, 16, (const void*) 0);
    gl_->EnableVertexAttribArray (0);
    gl_->VertexAttribPointer (1, 2, GL_FLOAT, GL_FALSE, 16, (const void*) 8);
    gl_->EnableVertexAttribArray (1);
    gl_->BindVertexArray (0);

    gl_->GenFramebuffers (1, &fbo_);

    setOutputSize (std::max (outWidth, 1), std::max (outHeight, 1));

    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (accumTex_[0]);
    if (gl_->CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        error = "FBO incomplete";
        gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
        shutdown();
        return false;
    }
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    if (! colorTransformGl_.initialize (const_cast<arbitgl::GlFuncs*> (gl_), error))
    {
        shutdown();
        return false;
    }
    return true;
}

unsigned FrameRenderer::createOutputTexture() const
{
    GLuint tex = 0;
    glGenTextures (1, &tex);
    glBindTexture (GL_TEXTURE_2D, tex);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    // HDR float intermediates (visuals-quality increment 1): every scratch /
    // composite / effect / feedback target carries linear values that may
    // exceed 1.0, so a later bloom/tonemap pass can extract highlights instead
    // of clamping them away at each ping-pong step. The final readback/present
    // path stays 8-bit (glReadPixels GL_UNSIGNED_BYTE / window blit), which
    // converts float→unorm8 with a [0,1] clamp — identical to the old RGBA8
    // result for in-range content, so the encoder path and [0,1] output are
    // unchanged. RGBA16F is core-renderable in GL 3.3. (Decoded video/image
    // INPUT textures stay RGBA8 — they are 8-bit source content, allocated
    // separately, not through this helper.)
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA16F, outW_, outH_, 0,
                  GL_RGBA, GL_HALF_FLOAT, nullptr);
    return tex;
}

unsigned FrameRenderer::cloneOutputTexture(unsigned source) const
{
    const unsigned target = createOutputTexture();
    if (source == 0 || target == 0)
        return target;

    GLuint readFramebuffer = 0;
    gl_->GenFramebuffers(1, &readFramebuffer);
    gl_->BindFramebuffer(GL_READ_FRAMEBUFFER, readFramebuffer);
    gl_->FramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                              GL_TEXTURE_2D, source, 0);
    if (gl_->CheckFramebufferStatus(GL_READ_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE)
    {
        glBindTexture(GL_TEXTURE_2D, target);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, outW_, outH_);
    }
    gl_->BindFramebuffer(GL_FRAMEBUFFER, fbo_);
    gl_->DeleteFramebuffers(1, &readFramebuffer);
    return target;
}

bool FrameRenderer::cloneRenderPassOutputs(
    const videowire::RenderPassOutputPublicationPtr& source,
    videowire::RenderPassOutputPublicationPtr& destination,
    std::string& error) const
{
    destination.reset();
    if (! source) return true;

    renderpassoutput::Description description;
    description.extent = source->outputs().extent();
    description.attachments.reserve(source->outputs().attachments().size());
    for (const auto& attachment : source->outputs().attachments())
        description.attachments.push_back({ attachment.output, attachment.format,
                                            attachment.colorSpace, attachment.extent });

    if (source->aovInspectionExecution().submitted)
        return videowire::makeAovInspectionPassPublication(
            source->sceneAovPayload(), source->aovInspectionPayload(), destination, error);
    if (source->sceneAovExecution().submitted)
        return videowire::makeSceneAovPassPublication(
            source->sceneAovPayload(), destination, error);
    if (source->motionAovExecution().submitted)
        return videowire::makeMotionAovPassPublication(
            description, source->motionAovClear(), destination, error);
    if (source->colorAovExecution().submitted)
        return videowire::makeColorAovPassPublication(
            description, source->colorAovClear(), destination, error);
    return videowire::makeRenderPassOutputPublication(description, destination, error);
}

FrameRenderer::FramePublicationCandidatePtr FrameRenderer::beginFramePublicationCandidate(
    std::uint64_t owner, std::uint64_t generation,
    const FramePublicationCandidatePtr& predecessor, std::string& error)
{
    error.clear();
    if (! ready() || owner == 0 || generation == 0)
    {
        error = "frame publication requires an initialized renderer and exact owner generation";
        return {};
    }
    if (predecessor && (predecessor->renderer != this || predecessor->discarded
                        || predecessor->owner != owner
                        || predecessor->generation + 1 != generation))
    {
        error = "frame publication predecessor does not belong to this renderer";
        return {};
    }
    auto candidate = std::make_shared<FramePublicationCandidate>();
    candidate->renderer = this;
    candidate->owner = owner;
    candidate->generation = generation;
    candidate->serial = nextFrameCandidateSerial_++;
    candidate->basePublicationGeneration = predecessor
        ? predecessor->basePublicationGeneration + 1 : publishedFrameGeneration_;
    candidate->predecessor = predecessor;
    const auto& sourceFeedback = predecessor ? predecessor->feedbackHistory : feedbackHistory_;
    const auto& sourceDelay = predecessor ? predecessor->frameDelayHistory : frameDelayHistory_;
    candidate->frameDelayHistorySerial = predecessor
        ? predecessor->frameDelayHistorySerial : frameDelayHistorySerial_;
    candidate->frameParity = predecessor ? predecessor->frameParity : frameParity_;
    const auto& sourceOutputs = predecessor ? predecessor->renderPassOutputs : renderPassOutputs_;
    if (! cloneRenderPassOutputs(sourceOutputs, candidate->renderPassOutputs, error))
        return {};
    candidate->release = [this](FramePublicationCandidate& value)
    {
        for (auto& item : value.feedbackHistory)
            for (auto& texture : item.second.tex)
                if (texture != 0) glDeleteTextures(1, &texture);
        for (auto& item : value.frameDelayHistory)
        {
            if (! item.second.ring.empty())
                glDeleteTextures(static_cast<GLsizei>(item.second.ring.size()), item.second.ring.data());
            if (item.second.output != 0) glDeleteTextures(1, &item.second.output);
        }
        value.feedbackHistory.clear();
        value.frameDelayHistory.clear();
        value.renderPassOutputs.reset();
    };
    for (const auto& item : sourceFeedback)
    {
        auto copy = item.second;
        copy.tex[0] = cloneOutputTexture(item.second.tex[0]);
        copy.tex[1] = cloneOutputTexture(item.second.tex[1]);
        candidate->feedbackHistory.emplace(item.first, std::move(copy));
    }
    for (const auto& item : sourceDelay)
    {
        auto copy = item.second;
        copy.ring.clear();
        copy.ring.reserve(item.second.ring.size());
        for (const auto texture : item.second.ring)
            copy.ring.push_back(cloneOutputTexture(texture));
        copy.output = cloneOutputTexture(item.second.output);
        candidate->frameDelayHistory.emplace(item.first, std::move(copy));
    }
    latestFrameCandidateSerial_[{ owner, generation }] = candidate->serial;
    frameCandidates_.push_back(candidate);
    activeFrameCandidate_ = candidate;
    return candidate;
}

bool FrameRenderer::validateFramePublicationCandidate(
    const FramePublicationCandidatePtr& candidate, std::string& error) const
{
    error.clear();
    const auto predecessor = candidate ? candidate->predecessor : FramePublicationCandidatePtr{};
    const auto latest = candidate
        ? latestFrameCandidateSerial_.find({ candidate->owner, candidate->generation })
        : latestFrameCandidateSerial_.end();
    if (! candidate || candidate->renderer != this || candidate->discarded || candidate->committed
        || latest == latestFrameCandidateSerial_.end()
        || latest->second != candidate->serial
        || (predecessor && ! predecessor->committed)
        || publishedFrameGeneration_ != candidate->basePublicationGeneration)
    {
        error = "frame publication owner or generation changed before atomic promotion";
        return false;
    }
    return true;
}

void FrameRenderer::promoteFramePublicationCandidate(
    const FramePublicationCandidatePtr& candidate) noexcept
{
    for (auto& item : feedbackHistory_)
        for (auto& texture : item.second.tex)
            if (texture != 0) glDeleteTextures(1, &texture);
    for (auto& item : frameDelayHistory_)
    {
        if (! item.second.ring.empty())
            glDeleteTextures(static_cast<GLsizei>(item.second.ring.size()), item.second.ring.data());
        if (item.second.output != 0) glDeleteTextures(1, &item.second.output);
    }
    feedbackHistory_ = std::move(candidate->feedbackHistory);
    frameDelayHistory_ = std::move(candidate->frameDelayHistory);
    frameDelayHistorySerial_ = candidate->frameDelayHistorySerial;
    frameParity_ = candidate->frameParity;
    renderPassOutputs_ = std::move(candidate->renderPassOutputs);
    publishedFrameOwner_ = candidate->owner;
    ++publishedFrameGeneration_;

    candidate->committed = true;
    candidate->release = {};
    if (activeFrameCandidate_ == candidate) activeFrameCandidate_.reset();
}

bool FrameRenderer::commitFramePublicationCandidate(
    const FramePublicationCandidatePtr& candidate, std::string& error)
{
    if (! validateFramePublicationCandidate(candidate, error)) return false;
    promoteFramePublicationCandidate(candidate);
    return true;
}

void FrameRenderer::discardFramePublicationCandidate(FramePublicationCandidatePtr& candidate) noexcept
{
    if (! candidate) return;
    candidate->discarded = true;
    if (candidate->release)
    {
        candidate->release(*candidate);
        candidate->release = {};
    }
    if (activeFrameCandidate_ == candidate) activeFrameCandidate_.reset();
    candidate.reset();
}

void FrameRenderer::activateFramePublicationCandidate(
    const FramePublicationCandidatePtr& candidate) noexcept
{
    activeFrameCandidate_ = candidate;
}

std::size_t FrameRenderer::liveFramePublicationCandidates() const noexcept
{
    return static_cast<std::size_t>(std::count_if(frameCandidates_.begin(), frameCandidates_.end(),
        [](const auto& value) { return ! value.expired(); }));
}

unsigned FrameRenderer::createColorTransformTexture() const
{
    GLuint tex = 0;
    glGenTextures (1, &tex);
    glBindTexture (GL_TEXTURE_2D, tex);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, outW_, outH_, 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    return tex;
}

void FrameRenderer::releaseTargets()
{
    for (auto& weak : frameCandidates_)
        if (auto candidate = weak.lock())
        {
            candidate->discarded = true;
            if (candidate->release)
            {
                candidate->release(*candidate);
                candidate->release = {};
            }
            candidate->renderer = nullptr;
        }
    activeFrameCandidate_.reset();
    frameCandidates_.clear();
    latestFrameCandidateSerial_.clear();
    GLuint texs[8] = { accumTex_[0], accumTex_[1], layerTexA_, layerTexB_,
                       fxTex_[0], fxTex_[1], transformInspectionTex_,
                       colorTransformTex_ };
    glDeleteTextures (8, texs);
    accumTex_[0] = accumTex_[1] = layerTexA_ = layerTexB_ = fxTex_[0] = fxTex_[1] = 0;
    transformInspectionTex_ = 0;
    colorTransformTex_ = 0;
    transformInspectionReady_ = false;
    if (hashPbo_[0] != 0 || hashPbo_[1] != 0)
    {
        gl_->DeleteBuffers (2, hashPbo_);
        hashPbo_[0] = hashPbo_[1] = 0;
    }
    hashPending_[0] = hashPending_[1] = false;

    if (exportPbo_[0] != 0 || exportPbo_[1] != 0)
    {
        gl_->DeleteBuffers (2, exportPbo_);
        exportPbo_[0] = exportPbo_[1] = 0;
    }
    exportPending_[0] = exportPending_[1] = false;
    exportPboBytes_ = 0;

    if (frameMapped_)
        endMappedReadback();
    if (framePbo_[0] != 0 || framePbo_[1] != 0)
    {
        gl_->DeleteBuffers (2, framePbo_);
        framePbo_[0] = framePbo_[1] = 0;
    }
    framePending_[0] = framePending_[1] = false;
    framePboW_[0] = framePboW_[1] = framePboH_[0] = framePboH_[1] = 0;

    // Scope target is fixed-size but lives in the same lifecycle: freed on
    // resize/shutdown, lazily recreated by the next readbackScope call.
    if (scopeTex_ != 0)
    {
        glDeleteTextures (1, &scopeTex_);
        scopeTex_ = 0;
    }
    if (scopePbo_[0] != 0 || scopePbo_[1] != 0)
    {
        gl_->DeleteBuffers (2, scopePbo_);
        scopePbo_[0] = scopePbo_[1] = 0;
    }
    scopePending_[0] = scopePending_[1] = false;

    // Feedback history is full-canvas and resolution-bound: free it on every
    // resize/shutdown so it is lazily recreated (cleared) at the new size —
    // which is exactly the canvas-resize reset rule (PINGPONG.md §3).
    for (auto& kv : feedbackHistory_)
        for (int b = 0; b < 2; ++b)
            if (kv.second.tex[b] != 0)
                glDeleteTextures (1, &kv.second.tex[b]);
    feedbackHistory_.clear();
    for (auto& kv : frameDelayHistory_)
    {
        if (! kv.second.ring.empty())
            glDeleteTextures(static_cast<GLsizei>(kv.second.ring.size()),
                             kv.second.ring.data());
        if (kv.second.output != 0)
            glDeleteTextures(1, &kv.second.output);
    }
    frameDelayHistory_.clear();
}

void FrameRenderer::setOutputSize (int outWidth, int outHeight)
{
    outWidth = std::max (outWidth, 1);
    outHeight = std::max (outHeight, 1);
    if (outWidth != outW_ || outHeight != outH_)
        renderPassOutputs_.reset();
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        outW_ = outWidth;
        outH_ = outHeight;
        if (metalRenderer_ != nullptr)
            metalRenderer_->setOutputSize (gl_, outW_, outH_);
        return;
    }
#endif
    if (gl_ == nullptr || (outWidth == outW_ && outHeight == outH_ && accumTex_[0] != 0))
        return;

    if (accumTex_[0] != 0)
        releaseTargets();

    outW_ = outWidth;
    outH_ = outHeight;
    accumTex_[0] = createOutputTexture();
    accumTex_[1] = createOutputTexture();
    layerTexA_   = createOutputTexture();
    layerTexB_   = createOutputTexture();
    fxTex_[0]    = createOutputTexture();
    fxTex_[1]    = createOutputTexture();
    colorTransformTex_ = createColorTransformTexture();

    // Hash readback region: the first ~1 MB of rows (full rows for stride
    // simplicity). Double-buffered PBOs sized to it.
    hashRows_ = std::min (outH_, std::max (1, (1 << 20) / std::max (outW_ * 4, 1)));
    hashBytes_ = hashRows_ * outW_ * 4;
    gl_->GenBuffers (2, hashPbo_);
    for (int i = 0; i < 2; ++i)
    {
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, hashPbo_[i]);
        gl_->BufferData (GL_PIXEL_PACK_BUFFER, hashBytes_, nullptr, GL_STREAM_READ);
    }
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    hashPending_[0] = hashPending_[1] = false;
    hashIdx_ = 0;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setOutputSize (gl_, outW_, outH_);
#endif
}

bool FrameRenderer::replaceRenderPassOutputs (
    const renderpassoutput::Description& description, std::string& error)
{
    auto& outputs = activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
    if (! ready())
    {
        error = "render-pass outputs require an initialized renderer";
        return false;
    }
    const renderpassoutput::Extent rendererExtent {
        static_cast<std::uint32_t> (outW_), static_cast<std::uint32_t> (outH_)
    };
    if (description.extent != rendererExtent)
    {
        error = "render-pass output extent does not match renderer output";
        return false;
    }
    return videowire::makeRenderPassOutputPublication (
        description, outputs, error);
}

const videowire::RenderPassOutputPublicationPtr& FrameRenderer::renderPassOutputs() const noexcept
{
    return activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
}

bool FrameRenderer::replaceColorAovPass (
    const renderpassoutput::Description& description, std::string& error)
{
    return replaceColorAovPass (
        description, arbitgpu::RenderPassColorAovClear {}, error);
}

bool FrameRenderer::prepareTemporalFeedbackPass (
    const videowire::TemporalFeedbackPass& pass, std::string& error) const
{
    error.clear();
    if (! ready() || metalOnly_ || gl_ == nullptr)
    {
        error = "native OpenGL temporal resource backend is unavailable";
        return false;
    }
    if (! visualtemporaloperation::validate(pass.payload, &error))
        return false;
    const bool frameDelay = pass.payload.mode == visualtemporaloperation::Mode::frameDelay;
    const bool feedback = pass.payload.mode == visualtemporaloperation::Mode::feedback;
    if ((! frameDelay && ! feedback) || (feedback && feedback_.prog == 0))
    {
        error = "native OpenGL temporal operation is unsupported";
        return false;
    }
    if (pass.clipId < 0 || pass.nodeId <= 0
        || pass.format != videotemporal::PixelFormat::rgba16f
        || pass.historyLength != pass.payload.historyLength)
    {
        error = "temporal resource pass description is malformed";
        return false;
    }
    if (pass.extent.width != static_cast<uint32_t>(outW_)
        || pass.extent.height != static_cast<uint32_t>(outH_))
    {
        error = "temporal feedback extent does not match the active compositor";
        return false;
    }

    const videotemporal::ResourceContract contract(
        frameDelay ? videotemporal::Mode::frameDelay : videotemporal::Mode::feedback,
        pass.format, pass.extent, pass.historyLength);
    videotemporal::ResourceLimits limits;
    limits.maximumImageDimension = static_cast<uint32_t>(
        std::max({ outW_, outH_, 1 }));
    limits.maximumHistoryLength = visualtemporaloperation::kMaximumHistoryLength;
    limits.maximumHistoryBytes = pass.footprint.historyBytes;
    limits.maximumTransientBytes = pass.footprint.transientBytes;
    videotemporal::ResourceFootprint verified;
    if (! videotemporal::admitResource(contract, limits, false, verified, error))
        return false;
    if (verified.retainedImages != pass.footprint.retainedImages
        || verified.transientImages != pass.footprint.transientImages
        || verified.bytesPerImage != pass.footprint.bytesPerImage
        || verified.historyBytes != pass.footprint.historyBytes
        || verified.transientBytes != pass.footprint.transientBytes)
    {
        error = "temporal resource footprint disagrees with admitted GPU resources";
        return false;
    }
    return true;
}

bool FrameRenderer::prepareMotionBlurPass (
    const videowire::TemporalSamplingExecution& execution,
    LayerDesc& layer, std::string& error) const
{
    const auto& payload = execution.pass.payload;
    error.clear();
    if (! ready() || metalOnly_ || gl_ == nullptr || motionBlur_.prog == 0)
    {
        error = "native OpenGL motion-blur composition is unavailable";
        return false;
    }
    if (! visualtemporalsampling::validate(payload, &error)
        || payload.mode != visualtemporalsampling::Mode::motionBlur)
    {
        if (error.empty()) error = "motion-blur sampling payload is malformed";
        return false;
    }
    if (execution.transition.action == visualtemporalsampling::TransitionAction::reject
        || execution.point.ownerId != execution.pass.clipId
        || execution.point.structuralRevision != execution.pass.structuralRevision
        || execution.point.width != execution.pass.width
        || execution.point.height != execution.pass.height)
    {
        error = "motion-blur temporal lifecycle does not match the admitted pass";
        return false;
    }
    const auto& outputs = renderPassOutputs();
    if (outputs == nullptr)
    {
        error = "motion blur requires an admitted Motion AOV publication";
        return false;
    }
    const auto* attachment = outputs->outputs().find(renderpassoutput::Output::Motion);
    const auto* resource = outputs->resource(renderpassoutput::Output::Motion);
    if (attachment == nullptr
        || attachment->format != renderpassoutput::PixelFormat::RG16Float
        || attachment->extent.width != static_cast<std::uint32_t>(outW_)
        || attachment->extent.height != static_cast<std::uint32_t>(outH_)
        || resource == nullptr || resource->textureView == 0
        || resource->textureView > std::numeric_limits<unsigned>::max())
    {
        error = "motion blur requires one compatible native RG16F Motion AOV";
        return false;
    }

    layer.graphMotionBlurActive = true;
    layer.graphMotionTexture = static_cast<unsigned>(resource->textureView);
    layer.graphMotionBlur = payload;
    return true;
}

bool FrameRenderer::replaceColorAovPass (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassColorAovClear& clear,
    std::string& error)
{
    auto& outputs = activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
    if (! ready())
    {
        error = "Color AOV pass requires an initialized renderer";
        return false;
    }
    const renderpassoutput::Extent rendererExtent {
        static_cast<std::uint32_t> (outW_), static_cast<std::uint32_t> (outH_)
    };
    if (description.extent != rendererExtent)
    {
        error = "Color AOV pass extent does not match renderer output";
        return false;
    }
    if (outputs != nullptr
        && outputs->colorAovExecution().submitted
        && outputs->colorAovExecution().submission != 0
        && outputs->colorAovExecution().error.empty()
        && outputs->colorAovClear().linearRgba == clear.linearRgba
        && description.attachments.size() == 1
        && outputs->outputs().extent() == description.extent
        && outputs->outputs().attachments().size() == 1)
    {
        const auto& admitted = outputs->outputs().attachments().front();
        const auto& requested = description.attachments.front();
        if (admitted.output == requested.output
            && admitted.format == requested.format
            && admitted.colorSpace == requested.colorSpace
            && admitted.extent == requested.extent)
        {
            error.clear();
            return true;
        }
    }
    return videowire::makeColorAovPassPublication (
        description, clear, outputs, error);
}

bool FrameRenderer::replaceMotionAovPass (
    const renderpassoutput::Description& description, std::string& error)
{
    return replaceMotionAovPass (
        description, arbitgpu::RenderPassMotionAovClear {}, error);
}

bool FrameRenderer::replaceMotionAovPass (
    const renderpassoutput::Description& description,
    const arbitgpu::RenderPassMotionAovClear& clear,
    std::string& error)
{
    auto& outputs = activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
    if (! ready())
    {
        error = "Motion AOV pass requires an initialized renderer";
        return false;
    }
    const renderpassoutput::Extent rendererExtent {
        static_cast<std::uint32_t> (outW_), static_cast<std::uint32_t> (outH_)
    };
    if (description.extent != rendererExtent)
    {
        error = "Motion AOV pass extent does not match renderer output";
        return false;
    }
    if (outputs != nullptr
        && outputs->motionAovExecution().submitted
        && outputs->motionAovExecution().submission != 0
        && outputs->motionAovExecution().error.empty()
        && outputs->motionAovClear().pixelDisplacement == clear.pixelDisplacement
        && description.attachments.size() == 1
        && outputs->outputs().extent() == description.extent
        && outputs->outputs().attachments().size() == 1)
    {
        const auto& admitted = outputs->outputs().attachments().front();
        const auto& requested = description.attachments.front();
        if (admitted.output == requested.output
            && admitted.format == requested.format
            && admitted.colorSpace == requested.colorSpace
            && admitted.extent == requested.extent)
        {
            error.clear();
            return true;
        }
    }
    return videowire::makeMotionAovPassPublication (
        description, clear, outputs, error);
}

bool FrameRenderer::replaceSceneAovPass (const sceneaov::Payload& payload,
                                         std::string& error)
{
    auto& outputs = activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
    return videowire::makeSceneAovPassPublication(payload, outputs, error);
}

bool FrameRenderer::replaceAovInspectionPass (const sceneaov::Payload& scenePayload,
                                              const aovinspection::Payload& payload,
                                              std::string& error)
{
    auto& outputs = activeFrameCandidate_ ? activeFrameCandidate_->renderPassOutputs : renderPassOutputs_;
    if (payload.extent != renderpassoutput::Extent {
            static_cast<std::uint32_t> (outW_), static_cast<std::uint32_t> (outH_) })
    {
        error = "AOV inspection extent does not match renderer output";
        return false;
    }
    if (outputs != nullptr
        && outputs->aovInspectionExecution().submitted
        && outputs->aovInspectionExecution().submission != 0
        && outputs->aovInspectionExecution().error.empty())
    {
        const auto& current = outputs->aovInspectionPayload();
        const auto currentScene = sceneaov::serialize (outputs->sceneAovPayload());
        const auto requestedScene = sceneaov::serialize (scenePayload);
        if (current.schemaVersion == payload.schemaVersion
            && current.source == payload.source
            && current.extent == payload.extent
            && current.depthNear == payload.depthNear
            && current.depthFar == payload.depthFar
            && current.invertDepth == payload.invertDepth
            && ! currentScene.empty() && currentScene == requestedScene)
        {
            error.clear();
            return true;
        }
    }
    return videowire::makeAovInspectionPassPublication (
        scenePayload, payload, outputs, error);
}

void FrameRenderer::setCanvas (int width, int height)
{
    canvasW_ = width;
    canvasH_ = height;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setCanvas (width, height);
#endif
}

void FrameRenderer::setPresentSize (int width, int height)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setPresentSize (width, height);
#endif
    if (gl_ == nullptr)
        return;
    if (width <= 0 || height <= 0)   // disable: applyCanvasFrame reverts to composite size
    {
        if (presentTex_ != 0) { glDeleteTextures (1, &presentTex_); presentTex_ = 0; }
        presentW_ = presentH_ = 0;
        return;
    }
    if (width == presentW_ && height == presentH_ && presentTex_ != 0)
        return;
    if (presentTex_ != 0) { glDeleteTextures (1, &presentTex_); presentTex_ = 0; }
    presentW_ = width;
    presentH_ = height;
    glGenTextures (1, &presentTex_);
    glBindTexture (GL_TEXTURE_2D, presentTex_);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA16F, presentW_, presentH_, 0,
                  GL_RGBA, GL_HALF_FLOAT, nullptr);
    glBindTexture (GL_TEXTURE_2D, 0);
}

void FrameRenderer::setView (float zoom, float panX, float panY)
{
    viewZoom_ = std::clamp (zoom, 0.02f, 32.0f);
    viewPanX_ = panX;
    viewPanY_ = panY;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setView (viewZoom_, viewPanX_, viewPanY_);
#endif
}

void FrameRenderer::viewMap (float& zx, float& zy, float& cx, float& cy) const
{
    if (canvasW_ <= 0 || canvasH_ <= 0)
    {
        // No canvas (export path): identity — output IS the canvas.
        zx = zy = 1.0f;
        cx = cy = 0.0f;
        return;
    }
    // Letterbox against the PRESENT (display) aspect when one is set — the
    // composite now renders at the canvas resolution (audit #16), so outW_/outH_
    // is the canvas itself and would give a no-op fit; the canvas must instead be
    // fitted into the panel/window the framed result is presented to.
    const int ow = presentW_ > 0 ? presentW_ : outW_;
    const int oh = presentW_ > 0 ? presentH_ : outH_;
    const float outAspect = oh > 0 ? (float) ow / (float) oh : 1.0f;
    const float canAspect = (float) canvasW_ / (float) canvasH_;
    // Fit the canvas inside the output (zoom 1 = exact fit, letterboxed).
    float ex = 1.0f, ey = 1.0f;
    if (canAspect > outAspect) ey = outAspect / canAspect;
    else                       ex = canAspect / outAspect;
    zx = viewZoom_ * ex;
    zy = viewZoom_ * ey;
    cx = viewPanX_;
    cy = viewPanY_;
}

void FrameRenderer::shutdown()
{
    renderPassOutputs_.reset();
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
    {
        metalRenderer_->shutdown (gl_);
        metalRenderer_.reset();
    }
    if (metalOnly_)
    {
        metalOnly_ = false;
        gl_ = nullptr;
        outW_ = outH_ = 0;
        return;
    }
#endif
    if (gl_ == nullptr)
        return;
    colorTransformGl_.shutdown();
    releaseTargets();
    if (presentTex_ != 0) { glDeleteTextures (1, &presentTex_); presentTex_ = 0; presentW_ = presentH_ = 0; }
    if (fbo_ != 0) { gl_->DeleteFramebuffers (1, &fbo_); fbo_ = 0; }
    if (vao_ != 0) { gl_->DeleteVertexArrays (1, &vao_); vao_ = 0; }
    if (vbo_ != 0) { gl_->DeleteBuffers (1, &vbo_); vbo_ = 0; }
    if (ebo_ != 0) { gl_->DeleteBuffers (1, &ebo_); ebo_ = 0; }
    for (unsigned* p : { &layer_.prog, &drawShape_.prog, &blend_.prog, &trans_.prog, &fx_.prog,
                  &blur_.prog, &sharpen_.prog, &productionFilter_.prog,
                  &lut_.prog, &blit_.prog, &frameBlend_.prog, &motionBlur_.prog,
                     &canvas_.prog, &preview_.prog, &feedback_.prog,
                     &bloomThresh_.prog, &bloomCombine_.prog })
        if (*p != 0) { gl_->DeleteProgram (*p); *p = 0; }
    for (auto& g : geom_)
        if (g.prog != 0) { gl_->DeleteProgram (g.prog); g.prog = 0; }
    // Per-clip shader generators own their own GL objects — free them while the
    // context is still current (before gl_ is nulled).
    for (auto& kv : shaderGens_)
        if (kv.second != nullptr)
            kv.second->shutdown (gl_);
    shaderGens_.clear();
    for (auto& kv : flatShaderBridges_)
        if (kv.second != nullptr)
            kv.second->shutdown (gl_);
    flatShaderBridges_.clear();
    shaderPlanClips_.clear();
    // Per-clip particle engines (P4) likewise own GL objects (SSBO/FBO/programs).
    for (auto& kv : particleGens_)
        if (kv.second != nullptr)
            kv.second->shutdown (gl_);
    particleGens_.clear();
    for (auto& [clipId, texture] : scoreTextures_)
    {
        (void) clipId;
        if (texture != 0) deleteTexture(texture);
    }
    scoreTextures_.clear();
    // Free any decoded image-INPUT textures not already released by clearClipShader.
    for (auto& [clipId, byName] : clipImageTex_)
        for (auto& [name, tex] : byName)
            if (tex != 0) deleteTexture (tex);
    clipImageTex_.clear();
    gl_ = nullptr;
    outW_ = outH_ = 0;
}

unsigned FrameRenderer::uploadRgba (const uint8_t* rgba, int width, int height,
                                    int strideBytes, unsigned existingTexture)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        unsigned handle = existingTexture;
        if (handle == 0)
        {
            handle = nextMetalHandle_++;
            if (nextMetalHandle_ == 0) nextMetalHandle_ = 0x80000000u;
        }
        if (metalRenderer_ != nullptr)
            metalRenderer_->uploadRgba (handle, rgba, width, height, strideBytes);
        return handle;
    }
#endif
    GLuint tex = existingTexture;
    if (tex == 0)
    {
        glGenTextures (1, &tex);
        glBindTexture (GL_TEXTURE_2D, tex);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    else
    {
        glBindTexture (GL_TEXTURE_2D, tex);
    }
    glPixelStorei (GL_UNPACK_ROW_LENGTH, strideBytes / 4);
    glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                  GL_RGBA, GL_UNSIGNED_BYTE, rgba);
    glPixelStorei (GL_UNPACK_ROW_LENGTH, 0);
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->uploadRgba (tex, rgba, width, height, strideBytes);
#endif
    return tex;
}

unsigned FrameRenderer::uploadR16 (const uint16_t* pixels, int width, int height,
                                   unsigned existingTexture)
{
    if (pixels == nullptr || width <= 0 || height <= 0 || width > 16384 || height > 16384)
        return 0;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        unsigned handle = existingTexture;
        if (handle == 0)
        {
            handle = nextMetalHandle_++;
            if (nextMetalHandle_ == 0) nextMetalHandle_ = 0x80000000u;
        }
        if (metalRenderer_ == nullptr
            || ! metalRenderer_->uploadR16 (handle, pixels, width, height))
            return 0;
        return handle;
    }
#endif
    GLuint tex = existingTexture;
    const size_t pixelCount = static_cast<size_t> (width) * static_cast<size_t> (height);
    std::vector<float> expanded (pixelCount);
    std::transform (pixels, pixels + pixelCount, expanded.begin(), [] (uint16_t value)
    {
        return static_cast<float> (value) / 65535.0f;
    });
    if (tex == 0) glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    GLint previousAlignment = 4;
    glGetIntegerv(GL_UNPACK_ALIGNMENT, &previousAlignment);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, width, height, 0, GL_RED,
                 GL_FLOAT, expanded.data());
    glPixelStorei(GL_UNPACK_ALIGNMENT, previousAlignment);
    return tex;
}

void FrameRenderer::deleteTexture (unsigned texture)
{
    if (texture != 0)
    {
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
        if (metalRenderer_ != nullptr)
            metalRenderer_->deleteTexture (texture);
        if (metalOnly_)
            return;
#endif
        GLuint t = texture;
        glDeleteTextures (1, &t);
    }
}

unsigned FrameRenderer::uploadLut3D (const float* rgbTriples, int size,
                                     unsigned existingTexture)
{
    if (rgbTriples == nullptr || size < 2)
        return existingTexture;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        unsigned handle = existingTexture;
        if (handle == 0)
        {
            handle = nextMetalHandle_++;
            if (nextMetalHandle_ == 0) nextMetalHandle_ = 0x80000000u;
        }
        if (metalRenderer_ != nullptr)
            metalRenderer_->uploadLut3D (handle, rgbTriples, size);
        return handle;
    }
#endif
    GLuint tex = existingTexture;
    if (tex == 0)
        glGenTextures (1, &tex);
    glBindTexture (GL_TEXTURE_3D, tex);
    glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri (GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
    // .cube data order (red fastest, then green, then blue) matches GL's
    // (s fastest, then t, then r) when s=R, t=G, r=B — upload as-is.
    gl_->TexImage3D (GL_TEXTURE_3D, 0, GL_RGB32F, size, size, size, 0,
                     GL_RGB, GL_FLOAT, rgbTriples);
    glBindTexture (GL_TEXTURE_3D, 0);
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->uploadLut3D (tex, rgbTriples, size);
#endif
    return tex;
}

void FrameRenderer::attachTarget (unsigned texture) const
{
    gl_->FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, texture, 0);
}

void FrameRenderer::drawQuad() const
{
    gl_->BindVertexArray (vao_);
    glDrawElements (GL_TRIANGLES, 6, GL_UNSIGNED_INT, nullptr);
}

void FrameRenderer::bindTexture (int unit, unsigned texture, int samplerLocation) const
{
    gl_->ActiveTexture (GL_TEXTURE0 + (GLenum) unit);
    glBindTexture (GL_TEXTURE_2D, texture);
    gl_->Uniform1i (samplerLocation, unit);
}

void FrameRenderer::setPassthrough (int transformLoc, int cropLoc) const
{
    // Identity transform + zero crop for fullscreen passes (the reference
    // relied on driver-zeroed uniforms here — compositor-spec §3.1 warning).
    gl_->UniformMatrix4fv (transformLoc, 1, GL_FALSE, kIdentity);
    gl_->Uniform4f (cropLoc, 0.0f, 0.0f, 0.0f, 0.0f);
}

unsigned FrameRenderer::runEffectChain (const LayerDesc& layer)
{
    auto& feedbackHistory = activeFrameCandidate_ ? activeFrameCandidate_->feedbackHistory : feedbackHistory_;
    auto& frameDelayHistory = activeFrameCandidate_ ? activeFrameCandidate_->frameDelayHistory : frameDelayHistory_;
    auto& frameDelayHistorySerial = activeFrameCandidate_
        ? activeFrameCandidate_->frameDelayHistorySerial : frameDelayHistorySerial_;
    if ((layer.effects == nullptr || layer.effectCount <= 0)
        && layer.lutTexture == 0 && ! layer.graphColorTransformActive
        && ! layer.graphMotionBlurActive && ! layer.graphTemporalActive)
        return layer.texture;

    int mask = 0;
    float values[FxUniformCount];
    std::memcpy (values, kFxNeutral, sizeof (values));
    float blurRadius = 0.0f;
    bool blurOn = false;
    float sharpenAmount = 0.0f;
    bool sharpenOn = false;

    // Routing follows videofx::kEffectDefs: params are stored in table order
    // (apply()/setEffects index them via effectParamIndex), so params[0] is
    // each effect's first table param and vignette's softness is params[1].
    // Two enabled effects of the same type: last slot wins (reference §7).
    for (int i = 0; layer.effects != nullptr && i < layer.effectCount; ++i)
    {
        const EffectSlotState& e = layer.effects[i];
        if (! e.enabled || e.type < 0 || e.type >= videofx::kEffectTypeCount)
            continue;
        mask |= videofx::kEffectBits[e.type];
        switch ((videofx::EffectType) e.type)
        {
            case videofx::EffectType::Brightness:    values[FxBrightness] = e.params[0]; break;
            case videofx::EffectType::Contrast:      values[FxContrast] = e.params[0]; break;
            case videofx::EffectType::Saturation:    values[FxSaturation] = e.params[0]; break;
            case videofx::EffectType::Hue:           values[FxHueShift] = e.params[0]; break;
            case videofx::EffectType::Exposure:      values[FxExposure] = e.params[0]; break;
            case videofx::EffectType::Gamma:         values[FxGamma] = e.params[0]; break;
            case videofx::EffectType::Blur:          blurRadius = e.params[0]; blurOn = true; break;
            case videofx::EffectType::Sharpen:       sharpenAmount = e.params[0]; sharpenOn = true; break;
            case videofx::EffectType::Vignette:      values[FxVignetteAmount] = e.params[0];
                                                     values[FxVignetteSoftness] = e.params[1]; break;
            case videofx::EffectType::Warm:          values[FxWarmth] = e.params[0]; break;
            case videofx::EffectType::Cool:          values[FxCoolness] = e.params[0]; break;
            case videofx::EffectType::Vintage:       values[FxVintage] = e.params[0]; break;
            case videofx::EffectType::Sepia:         values[FxSepia] = e.params[0]; break;
            case videofx::EffectType::BlackAndWhite: values[FxBW] = e.params[0]; break;
            case videofx::EffectType::Invert:        values[FxInvert] = e.params[0]; break;
            case videofx::EffectType::Posterize:     values[FxPosterize] = e.params[0]; break;
            case videofx::EffectType::Noise:         values[FxNoise] = e.params[0]; break;
            case videofx::EffectType::ChromaKey:     for (int k = 0; k < 6; ++k)
                                                         values[FxKeyR + k] = e.params[k];
                                                     break;
            case videofx::EffectType::LumaKey:       for (int k = 0; k < 4; ++k)
                                                         values[FxLumaLow + k] = e.params[k];
                                                     break;
            case videofx::EffectType::ColorWheels:   for (int k = 0; k < 9; ++k)
                                                         values[FxLiftR + k] = e.params[k];
                                                     break;
            default: break;
        }
    }
    if (layer.graphKeyCleanupActive)
    {
        values[FxKeyChoke] = layer.graphKeyCleanupChoke;
        values[FxKeyFeather] = layer.graphKeyCleanupFeather;
        values[FxKeyEdgeR] = layer.graphKeyCleanupEdgeR;
        values[FxKeyEdgeG] = layer.graphKeyCleanupEdgeG;
        values[FxKeyEdgeB] = layer.graphKeyCleanupEdgeB;
        values[FxKeyEdgeAmount] = layer.graphKeyCleanupEdgeAmount;
        values[FxKeyMatteView] = layer.graphKeyCleanupMatteView ? 1.0f : 0.0f;
        values[FxKeyTexelX] = 1.0f / (float) std::max (outW_, 1);
        values[FxKeyTexelY] = 1.0f / (float) std::max (outH_, 1);
    }

    if (mask == 0 && layer.lutTexture == 0 && ! layer.graphColorTransformActive
        && ! layer.graphMotionBlurActive && ! layer.graphTemporalActive)
        return layer.texture;

    unsigned cur = layer.texture;
    auto nextFx = [&] (unsigned curTex) { return curTex == fxTex_[0] ? fxTex_[1] : fxTex_[0]; };

    if (layer.graphMotionBlurActive)
    {
        if (layer.graphMotionTexture == 0
            || ! visualtemporalsampling::validate(layer.graphMotionBlur)
            || layer.graphMotionBlur.mode != visualtemporalsampling::Mode::motionBlur)
        {
            lastError_ = "motion blur requires one admitted native RG16F Motion AOV";
            return 0;
        }
        const unsigned target = nextFx(cur);
        attachTarget(target);
        gl_->UseProgram(motionBlur_.prog);
        setPassthrough(motionBlur_.uTransform, motionBlur_.uCrop);
        bindTexture(0, cur, motionBlur_.uTexture);
        bindTexture(1, layer.graphMotionTexture, motionBlur_.uMotion);
        gl_->Uniform2f(motionBlur_.uResolution, (float) outW_, (float) outH_);
        gl_->Uniform1i(motionBlur_.uSampleCount,
                       (int) layer.graphMotionBlur.sampleCount);
        gl_->Uniform1f(motionBlur_.uShutterScale,
                       layer.graphMotionBlur.motionScale
                           * layer.graphMotionBlur.shutterAngleDegrees / 360.0f);
        drawQuad();
        cur = target;
    }

    if (layer.graphColorTransformActive)
    {
        auto description = layer.graphColorTransform;
        const colortransform::Extent dispatchExtent {
            static_cast<std::uint32_t>(outW_), static_cast<std::uint32_t>(outH_) };
        description.input.extent = dispatchExtent;
        description.output.extent = dispatchExtent;
        colortransform::AdmissionFailure failure = colortransform::AdmissionFailure::None;
        const auto admitted = colortransform::admit(
            description, colortransform::BackendCapability::NativeGpu, failure);
        const unsigned target = colorTransformTex_;
        std::string transformError;
        if (target == 0 || ! admitted || ! colorTransformGl_.render(
                cur, target, outW_, outH_, *admitted, transformError))
        {
            lastError_ = target == 0 ? "color transform RGBA8 target is unavailable"
                : admitted ? transformError
                : std::string("color transform admission failed: ")
                    + std::string(colortransform::token(failure));
            return 0;
        }
        cur = target;
    }

    const int blurBits = videofx::kEffectBits[(int) videofx::EffectType::Blur]
                       | videofx::kEffectBits[(int) videofx::EffectType::Sharpen];
    if ((mask & ~blurBits) != 0)
    {
        // Single color pass; all value uniforms uploaded every pass, the
        // bitmask is the real switch (effects-spec §8).
        const unsigned target = nextFx(cur);
        attachTarget (target);
        gl_->UseProgram (fx_.prog);
        setPassthrough (fx_.uTransform, fx_.uCrop);
        bindTexture (0, cur, fx_.uTexture);
        gl_->Uniform1f (fx_.uTime, (float) layer.timeSec);
        gl_->Uniform1i (fx_.uEffectMask, mask);
        for (int i = 0; i < FxUniformCount; ++i)
            gl_->Uniform1f (fx_.uValue[i], values[i]);
        drawQuad();
        cur = target;
    }

    if (blurOn && blurRadius > 0.0f)
    {
        const float stepX = 1.0f / (float) outW_;
        const float stepY = 1.0f / (float) outH_;
        for (int pass = 0; pass < 2; ++pass)
        {
            const unsigned target = nextFx (cur);
            attachTarget (target);
            gl_->UseProgram (blur_.prog);
            setPassthrough (blur_.uTransform, blur_.uCrop);
            bindTexture (0, cur, blur_.uTexture);
            gl_->Uniform2f (blur_.uTexelStep, pass == 0 ? stepX : 0.0f,
                                              pass == 0 ? 0.0f : stepY);
            gl_->Uniform1f (blur_.uRadius, blurRadius);
            drawQuad();
            cur = target;
        }
    }

    if (sharpenOn && sharpenAmount > 0.0f)
    {
        const unsigned target = nextFx (cur);
        attachTarget (target);
        gl_->UseProgram (sharpen_.prog);
        setPassthrough (sharpen_.uTransform, sharpen_.uCrop);
        bindTexture (0, cur, sharpen_.uTexture);
        gl_->Uniform2f (sharpen_.uTexelSize, 1.0f / (float) outW_, 1.0f / (float) outH_);
        gl_->Uniform1f (sharpen_.uAmount, sharpenAmount);
        drawQuad();
        cur = target;
    }

    // 3D LUT pass — last in the per-layer color chain (Arbit extension).
    if (layer.lutTexture != 0 && layer.lutSize >= 2)
    {
        const unsigned target = nextFx (cur);
        attachTarget (target);
        gl_->UseProgram (lut_.prog);
        setPassthrough (lut_.uTransform, lut_.uCrop);
        bindTexture (0, cur, lut_.uTexture);
        gl_->ActiveTexture (GL_TEXTURE1);
        glBindTexture (GL_TEXTURE_3D, layer.lutTexture);
        gl_->Uniform1i (lut_.uLut, 1);
        gl_->Uniform1f (lut_.uLutSize, (float) layer.lutSize);
        drawQuad();
        gl_->ActiveTexture (GL_TEXTURE1);
        glBindTexture (GL_TEXTURE_3D, 0);
        cur = target;
    }

    // Geometric / UV effect passes (Arbit extension, Jun 2026) — per-slot
    // ordered passes applied AFTER the color/blur/sharpen/LUT chain, so they
    // distort the graded result ("spiral, then kaleidoscope it"). Each reads
    // the previous texture and ping-pongs into the other fx target. A geometric
    // bit is never a blur bit, so when any geometric effect is enabled the
    // color pass above already ran (an identity passthrough when only geometric
    // effects are active) — `cur` is therefore already an fx ping-pong texture.
    for (int i = 0; layer.effects != nullptr && i < layer.effectCount; ++i)
    {
        const EffectSlotState& e = layer.effects[i];
        if (! e.enabled)
            continue;
        const int gi = geomIndexForType (e.type);
        if (gi < 0)
            continue;
        const GeomProg& g = geom_[gi];
        if (g.prog == 0)
            continue;
        const unsigned target = nextFx (cur);
        attachTarget (target);
        gl_->UseProgram (g.prog);
        setPassthrough (g.uTransform, g.uCrop);
        bindTexture (0, cur, g.uTexture);
        gl_->Uniform2f (g.uResolution, (float) outW_, (float) outH_);
        gl_->Uniform1f (g.uTime, (float) layer.timeSec);
        const videofx::EffectDef* def = videofx::effectDefFor (e.type);
        const int pc = def != nullptr ? def->paramCount : 0;
        for (int j = 0; j < pc && j < videofx::kMaxEffectParams; ++j)
            gl_->Uniform1f (g.uParam[j], e.params[j]);
        drawQuad();
        cur = target;
    }

    if (layer.graphTemporalActive
        && layer.graphTemporalPayload.mode == visualtemporaloperation::Mode::frameDelay)
    {
        const auto& payload = layer.graphTemporalPayload;
        if (! visualtemporaloperation::validate(payload) || layer.clipId < 0
            || layer.graphTemporalNodeId <= 0 || payload.historyLength == 0)
        {
            lastError_ = "frame delay requires one admitted RGBA16F history ring";
            return 0;
        }
        const auto key = std::make_pair(layer.clipId, layer.graphTemporalNodeId);
        auto historyIt = frameDelayHistory.find(key);
        if (historyIt == frameDelayHistory.end())
        {
            const auto releaseHistory = [] (FrameDelayHistory& stale)
            {
                if (! stale.ring.empty())
                    glDeleteTextures(static_cast<GLsizei>(stale.ring.size()), stale.ring.data());
                if (stale.output != 0) glDeleteTextures(1, &stale.output);
            };
            for (auto it = frameDelayHistory.begin(); it != frameDelayHistory.end();)
            {
                if (it->first.first != layer.clipId)
                {
                    ++it;
                    continue;
                }
                releaseHistory(it->second);
                it = frameDelayHistory.erase(it);
            }
            if (frameDelayHistory.size() >= kMaximumFrameDelayHistories)
            {
                const auto oldest = std::min_element(
                    frameDelayHistory.begin(), frameDelayHistory.end(),
                    [] (const auto& lhs, const auto& rhs)
                    {
                        return lhs.second.lastUseSerial < rhs.second.lastUseSerial;
                    });
                releaseHistory(oldest->second);
                frameDelayHistory.erase(oldest);
            }
            historyIt = frameDelayHistory.emplace(key, FrameDelayHistory {}).first;
        }
        auto& history = historyIt->second;
        history.lastUseSerial = ++frameDelayHistorySerial;
        const bool reset = layer.feedbackHistoryReset
            || history.structuralRevision != layer.visualPlanStructuralRevision
            || history.w != outW_ || history.h != outH_
            || history.ring.size() != payload.historyLength || history.output == 0;
        if (reset)
        {
            if (! history.ring.empty())
                glDeleteTextures(static_cast<GLsizei>(history.ring.size()), history.ring.data());
            if (history.output != 0) glDeleteTextures(1, &history.output);
            history = {};
            history.ring.resize(payload.historyLength);
            for (auto& texture : history.ring)
            {
                texture = createOutputTexture();
                attachTarget(texture);
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
            }
            history.output = createOutputTexture();
            attachTarget(history.output);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            history.structuralRevision = layer.visualPlanStructuralRevision;
            history.w = outW_;
            history.h = outH_;
            history.lastUseSerial = frameDelayHistorySerial;
        }
        if (! layer.feedbackHistoryHold)
        {
            attachTarget(history.output);
            glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            if (history.filled == history.ring.size())
            {
                gl_->UseProgram(blit_.prog);
                bindTexture(0, history.ring[history.cursor], blit_.uTexture);
                drawQuad();
            }
            attachTarget(history.ring[history.cursor]);
            gl_->UseProgram(blit_.prog);
            bindTexture(0, cur, blit_.uTexture);
            drawQuad();
            history.cursor = (history.cursor + 1u) % static_cast<uint32_t>(history.ring.size());
            history.filled = std::min<uint32_t>(history.filled + 1u,
                                                static_cast<uint32_t>(history.ring.size()));
        }
        cur = history.output;
    }

    // Cross-frame feedback trail (Arbit extension, Jun 2026) — the one
    // history-dependent pass. Blends the current result over a decayed copy of
    // THIS clip's previous-frame output, held in a persistent double-buffered
    // texture (PINGPONG.md §2). Runs LAST; a second FeedbackTrail slot is
    // ignored (v1 cap of one per clip). Parity: a fresh renderer (export) seeds
    // cleared history, so full-clip exports match live linear playback; only
    // scrubbed/seeked trails may differ (documented, master-plan principle 1).
    int fbSlot = -1;
    for (int i = 0; layer.effects != nullptr && i < layer.effectCount; ++i)
        if (layer.effects[i].enabled
            && layer.effects[i].type == (int) videofx::EffectType::FeedbackTrail)
        { fbSlot = i; break; }
    if (fbSlot >= 0 && feedback_.prog != 0)
    {
        const int key = (layer.clipId << 8) | (fbSlot & 0xFF);
        FeedbackHistory& hist = feedbackHistory[key];
        if (layer.feedbackHistoryReset || hist.tex[0] == 0 || hist.tex[1] == 0
            || hist.w != outW_ || hist.h != outH_)
        {
            // Clip start / seek / revision / resize: (re)allocate cleared.
            for (int b = 0; b < 2; ++b)
            {
                if (hist.tex[b] != 0) glDeleteTextures (1, &hist.tex[b]);
                hist.tex[b] = createOutputTexture();
                attachTarget (hist.tex[b]);
                glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
                glClear (GL_COLOR_BUFFER_BIT);
            }
            hist.w = outW_;
            hist.h = outH_;
            hist.current = 0;
            hist.ready = false;
        }
        if (layer.feedbackHistoryHold && hist.ready)
        {
            cur = hist.tex[hist.current];
        }
        else
        {
            const unsigned readIndex = hist.ready ? hist.current : 0;
            const unsigned writeIndex = hist.ready ? (hist.current ^ 1u) : 1u;
            const unsigned readTex = hist.tex[readIndex];
            const unsigned writeTex = hist.tex[writeIndex];
            attachTarget (writeTex);
        gl_->UseProgram (feedback_.prog);
        setPassthrough (feedback_.uTransform, feedback_.uCrop);
        bindTexture (0, cur, feedback_.uTexture);
        bindTexture (1, readTex, feedback_.uFeedbackTex);
        const videofx::EffectDef* def = videofx::effectDefFor ((int) videofx::EffectType::FeedbackTrail);
        const EffectSlotState& fe = layer.effects[fbSlot];
        const int pc = def != nullptr ? def->paramCount : 0;
        for (int j = 0; j < pc && j < videofx::kMaxEffectParams; ++j)
            gl_->Uniform1f (feedback_.uParam[j], fe.params[j]);
        drawQuad();
        gl_->ActiveTexture (GL_TEXTURE0);   // leave unit 0 active for later passes
        cur = writeTex;
        hist.current = writeIndex;
        hist.ready = true;
        }
    }

    return cur;
}

void FrameRenderer::drawLayerGeometry (unsigned sourceTexture, const LayerDesc& layer,
                                       unsigned targetTexture, float opacity)
{
    attachTarget (targetTexture);
    glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
    glClear (GL_COLOR_BUFFER_BIT);

    // Letterbox fit against the CANVAS aspect (the project's export frame;
    // falls back to the output aspect when no canvas is set — the export
    // path, where output == canvas). Then the user transform, then the
    // viewport-only view transform mapping canvas NDC -> output NDC.
    const float outAspect = outH_ > 0 ? (float) outW_ / (float) outH_ : 1.0f;
    const float refAspect = (canvasW_ > 0 && canvasH_ > 0)
        ? (float) canvasW_ / (float) canvasH_ : outAspect;
    const float vidAspect = layer.texHeight > 0
        ? (float) layer.texWidth / (float) layer.texHeight : 1.0f;
    float lbx = 1.0f, lby = 1.0f;
    if (vidAspect > refAspect) lby = refAspect / vidAspect;
    else                       lbx = vidAspect / refAspect;

    float m[16];
    buildTransform (layer.translateX, layer.translateY, layer.rotationDeg,
                    layer.scale * lbx, layer.scale * lby, m);
    // Final = V · M with V = T(pan) · S(zoom·fit) (column-major 2D affine).
    float zx, zy, cx, cy;
    viewMap (zx, zy, cx, cy);
    m[0] *= zx; m[1] *= zy; m[4] *= zx; m[5] *= zy;
    m[12] = m[12] * zx + cx;
    m[13] = m[13] * zy + cy;

    gl_->UseProgram (layer_.prog);
    gl_->UniformMatrix4fv (layer_.uTransform, 1, GL_FALSE, m);
    gl_->Uniform1i (layer_.uCornerPin, layer.cornerPin ? 1 : 0);
    for (int i = 0; i < 4; ++i)
        gl_->Uniform2f (layer_.uCorners + i, layer.corners[(size_t) i * 2],
                       layer.corners[(size_t) i * 2 + 1]);
    gl_->Uniform4f (layer_.uCrop, layer.cropLeft, layer.cropRight,
                    layer.cropTop, layer.cropBottom);
    gl_->Uniform1f (layer_.uOpacity, opacity);
    gl_->Uniform1i (layer_.uBlendMode, layer.blendMode); // unused by shader (parity)
    // Shape mask (Arbit extension): multiplies the layer alpha in RawUV
    // (displayed-frame) space.
    gl_->Uniform1i (layer_.uMaskType, layer.maskType);
    gl_->Uniform4f (layer_.uMaskRect, layer.maskCx, layer.maskCy,
                    layer.maskW, layer.maskH);
    gl_->Uniform1f (layer_.uMaskFeather, layer.maskFeather);
    gl_->Uniform1i (layer_.uMaskInvert, layer.maskInvert ? 1 : 0);
    gl_->Uniform4f (layer_.uPathRect, layer.pathMatteCx, layer.pathMatteCy,
                    layer.pathMatteEllipse ? -layer.pathMatteW : layer.pathMatteW,
                    layer.pathMatteH);
    gl_->Uniform4f (layer_.uPathRectB, layer.pathMatte2Cx, layer.pathMatte2Cy,
                    layer.pathMatteSecondaryEllipse ? -layer.pathMatte2W : layer.pathMatte2W,
                    layer.pathMatte2H);
    gl_->Uniform1i (layer_.uPathOperation,
                    layer.pathMatte ? std::clamp(layer.pathMatteOperation, 0, 3) : -1);
    gl_->Uniform1i (layer_.uPathInvert, layer.pathMatteInvert ? 1 : 0);
    const bool matte = layer.matteApply && layer.matteTexture != 0
        && layer.matteWidth > 0 && layer.matteHeight > 0;
    gl_->Uniform1i (layer_.uMatteApply, matte ? 1 : 0);
    gl_->Uniform2f (layer_.uMatteTexel, matte ? 1.0f / layer.matteWidth : 0.0f,
                    matte ? 1.0f / layer.matteHeight : 0.0f);
    gl_->Uniform4f (layer_.uMatteRefine, layer.matteBlack, layer.matteWhite,
                    layer.matteErodeDilate, layer.matteFeather);
    gl_->Uniform1f (layer_.uMatteChoke, layer.matteChoke);
    gl_->Uniform1i (layer_.uMatteInvert, layer.matteInvert ? 1 : 0);
    bindTexture (1, matte ? layer.matteTexture : sourceTexture, layer_.uMatteTexture);
    gl_->Uniform1i(layer_.uMatteCombineMode, layer.matteCombineMode);
    bindTexture(3, matte && layer.matteCombineMode >= 0 ? layer.matteTextureB : sourceTexture,
                layer_.uMatteTextureB);
    const int depthEffect = layer.depthEffect != 0 ? layer.depthEffect : (layer.depthFog ? 1 : 0);
    const bool depthFog = depthEffect != 0 && layer.depthTexture != 0
        && layer.depthWidth > 0 && layer.depthHeight > 0;
    gl_->Uniform1i(layer_.uDepthFog, depthFog ? depthEffect : 0);
    if (depthEffect <= 1) {
        gl_->Uniform3f(layer_.uFogRangeDensity, layer.fogNear, layer.fogFar, layer.fogDensity);
        gl_->Uniform4f(layer_.uFogColor, layer.fogRed, layer.fogGreen, layer.fogBlue, layer.fogAlpha);
    } else {
        gl_->Uniform3f(layer_.uFogRangeDensity, layer.depthParam0, layer.depthParam1, layer.depthParam2);
        gl_->Uniform4f(layer_.uFogColor, layer.depthColorRed, layer.depthColorGreen, layer.depthColorBlue, 1.0f);
    }
    bindTexture(2, depthFog ? layer.depthTexture : sourceTexture, layer_.uDepthTexture);
    bindTexture (0, sourceTexture, layer_.uTexture);
    drawQuad();
}

unsigned FrameRenderer::buildLayerFrame (const LayerDesc& layer, float& blendOpacityOut)
{
    if (layer.shaderOperationPlan != nullptr && !layer.shaderOperationPlan->operations.empty())
    {
        std::map<int, unsigned> resources;
        for (const auto& operation : layer.shaderOperationPlan->operations)
            for (std::size_t input = 0; input < operation.inputCount; ++input)
                if (resources.count(operation.inputNodeIds[input]) == 0)
                {
                    const LayerDesc* source = input == 0
                        && operation.kind == videowire::ShaderOperationKind::transition
                        ? layer.fromLayer : &layer;
                    if (source != nullptr && source->texture != 0)
                        resources[operation.inputNodeIds[input]] = source->texture;
                }
        unsigned result = 0;
        const bool executed = videowire::visitOrderedShaderOperationsOnce(
            layer.shaderOperationPlan, layer.shaderOperationParameters,
            [&](const auto& operation, const auto& values)
        {
            const FlatShaderKey key { layer.clipId, layer.visualPlanStructuralRevision,
                                      layer.shaderOperationPlan->digest,
                                      operation.nodeId, operation.payload.sourceSha256 };
            const auto bridge = flatShaderBridges_.find(key);
            if (bridge == flatShaderBridges_.end() || bridge->second == nullptr) return false;
            std::map<std::string, unsigned> images;
            if (operation.kind == videowire::ShaderOperationKind::filter)
                images["inputImage"] = resources[operation.inputNodeIds[0]];
            else if (operation.kind == videowire::ShaderOperationKind::transition)
            {
                images["startImage"] = resources[operation.inputNodeIds[0]];
                images["endImage"] = resources[operation.inputNodeIds[1]];
            }
            if (std::any_of(images.begin(), images.end(),
                            [] (const auto& image) { return image.second == 0; }))
                return false;
            result = bridge->second->render(gl_, layer.shaderClock, outW_, outH_,
                layer.audioPresent ? &layer.audioFeatures : nullptr,
                canonicalblockc::valid(layer.canonicalBlockCFrame)
                    ? layer.canonicalBlockCFrame.get() : nullptr,
                &values, &images);
            if (result == 0) return false;
            resources[operation.nodeId] = result;
            resources[operation.outputNodeId] = result;
            return true;
        });
        if (!executed) return 0;
        drawLayerGeometry(result, layer, layerTexB_, 1.0f);
        blendOpacityOut = layer.opacity;
        return layerTexB_;
    }

    if (layer.curatedShaderTransition != nullptr)
    {
        if (layer.fromLayer == nullptr || layer.fromLayer->texture == 0)
            return 0;

        const unsigned primary = runEffectChain (layer);
        if (primary == 0) return 0;
        drawLayerGeometry (primary, layer, layerTexB_, layer.opacity);
        const unsigned secondary = runEffectChain (*layer.fromLayer);
        if (secondary == 0) return 0;
        drawLayerGeometry (secondary, *layer.fromLayer, layerTexA_,
                           layer.fromLayer->opacity);

        const auto& operation = *layer.curatedShaderTransition;
        const FlatShaderKey key { layer.clipId, layer.visualPlanStructuralRevision, {},
                                  operation.nodeId, operation.payload.sourceSha256 };
        const auto bridge = flatShaderBridges_.find(key);
        if (bridge == flatShaderBridges_.end() || bridge->second == nullptr)
            return 0;

        auto values = operation.parameterValues;
        values["progress"] = operation.evaluatedProgress();
        const std::map<std::string, unsigned> images {
            { "startImage", layerTexA_ }, { "endImage", layerTexB_ }
        };
        const unsigned result = bridge->second->render(
            gl_, layer.shaderClock, outW_, outH_,
            layer.audioPresent ? &layer.audioFeatures : nullptr,
            canonicalblockc::valid(layer.canonicalBlockCFrame)
                ? layer.canonicalBlockCFrame.get() : nullptr,
            &values, &images);
        if (result == 0) return 0;
        blendOpacityOut = 1.0f;
        return result;
    }

    const int shaderType = transitionShaderType (layer.transitionType);

    if (shaderType < 0)
    {
        // Plain layer: effects, then geometry into the per-layer frame.
        // Opacity is deferred to the blend pass (alpha = front.a * uOpacity,
        // compositor-spec §5.2), so the frame keeps the texture's own alpha.
        const unsigned processed = runEffectChain (layer);
        if (processed == 0) return 0;
        drawLayerGeometry (processed, layer, layerTexB_, 1.0f);
        blendOpacityOut = layer.opacity;
        return layerTexB_;
    }

    // Transition: bake each side's opacity into its own frame, then blend
    // the transition result at opacity 1.
    const unsigned processedB = runEffectChain (layer);
    if (processedB == 0) return 0;
    drawLayerGeometry (processedB, layer, layerTexB_, layer.opacity);

    attachTarget (layerTexA_);
    glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
    glClear (GL_COLOR_BUFFER_BIT);
    if (layer.fromLayer != nullptr && layer.fromLayer->texture != 0)
    {
        const unsigned processedA = runEffectChain (*layer.fromLayer);
        if (processedA == 0) return 0;
        drawLayerGeometry (processedA, *layer.fromLayer, layerTexA_,
                           layer.fromLayer->opacity);
    }

    attachTarget (fxTex_[0]);
    gl_->UseProgram (trans_.prog);
    setPassthrough (trans_.uTransform, trans_.uCrop);
    bindTexture (0, layerTexA_, trans_.uFromTex);
    bindTexture (1, layerTexB_, trans_.uToTex);
    gl_->Uniform1f (trans_.uProgress,
                    std::clamp (layer.transitionProgress, 0.0f, 1.0f));
    gl_->Uniform1i (trans_.uType, shaderType);
    drawQuad();

    blendOpacityOut = 1.0f;
    return fxTex_[0];
}

void FrameRenderer::blendOnto (unsigned frontTexture, unsigned backTexture,
                               unsigned targetTexture, float opacity, int blendMode)
{
    attachTarget (targetTexture);
    gl_->UseProgram (blend_.prog);
    setPassthrough (blend_.uTransform, blend_.uCrop);
    bindTexture (0, backTexture, blend_.uBackTex);
    bindTexture (1, frontTexture, blend_.uFrontTex);
    gl_->Uniform1f (blend_.uOpacity, opacity);
    gl_->Uniform1i (blend_.uBlendMode, blendMode);
    drawQuad();
}

unsigned FrameRenderer::frameBlendInto (unsigned texA, unsigned texB, unsigned outTex,
                                        int width, int height, float mix)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        unsigned handle = outTex;
        if (handle == 0)
        {
            handle = nextMetalHandle_++;
            if (nextMetalHandle_ == 0) nextMetalHandle_ = 0x80000000u;
        }
        if (metalRenderer_ != nullptr)
            metalRenderer_->setFrameBlend (handle, texA, texB, width, height, mix);
        return handle;
    }
#endif
    GLuint tex = outTex;
    if (tex == 0)
    {
        glGenTextures (1, &tex);
        glBindTexture (GL_TEXTURE_2D, tex);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    }
    // decodeLayer runs before renderComposite, which re-binds fbo_ + resets the
    // viewport/attachment, so this pass owns those without saving/restoring.
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (tex);
    glViewport (0, 0, width, height);
    glDisable (GL_BLEND);   // mix is done in-shader (matches renderComposite)
    gl_->UseProgram (frameBlend_.prog);
    setPassthrough (frameBlend_.uTransform, frameBlend_.uCrop);
    bindTexture (0, texA, frameBlend_.uTexA);
    bindTexture (1, texB, frameBlend_.uTexB);
    gl_->Uniform1f (frameBlend_.uMix, mix);
    drawQuad();
    gl_->ActiveTexture (GL_TEXTURE0);   // leave unit 0 active for later passes
    return tex;
}

void FrameRenderer::drawImageOverlay (const ImageLayerDesc& overlay, unsigned targetTexture)
{
    attachTarget (targetTexture);
    glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
    glClear (GL_COLOR_BUFFER_BIT);

    // Natural pixel size: the fullscreen quad scaled to width/height of the
    // overlay texture in CANVAS pixels (output pixels when no canvas is set
    // — the export path), then offset by posX/posY (same sign convention as
    // clip translate: +posY moves down on screen), then the view transform.
    // Sizing against the canvas keeps text size identical between the
    // viewport and an export at canvas resolution.
    const float refW = canvasW_ > 0 ? (float) canvasW_ : (float) outW_;
    const float refH = canvasH_ > 0 ? (float) canvasH_ : (float) outH_;
    float m[16];
    buildTransform (overlay.posX, overlay.posY, 0.0f,
                    (float) overlay.width / refW,
                    (float) overlay.height / refH, m);
    float zx, zy, cx, cy;
    viewMap (zx, zy, cx, cy);
    m[0] *= zx; m[1] *= zy; m[4] *= zx; m[5] *= zy;
    m[12] = m[12] * zx + cx;
    m[13] = m[13] * zy + cy;

    gl_->UseProgram (layer_.prog);
    gl_->UniformMatrix4fv (layer_.uTransform, 1, GL_FALSE, m);
    gl_->Uniform1i (layer_.uCornerPin, 0);
    gl_->Uniform4f (layer_.uCrop, 0.0f, 0.0f, 0.0f, 0.0f);
    gl_->Uniform1f (layer_.uOpacity, 1.0f); // opacity applied at blend
    gl_->Uniform1i (layer_.uBlendMode, 0);
    gl_->Uniform1i (layer_.uMaskType, 0);   // overlays are never shape-masked
    gl_->Uniform1i (layer_.uMaskInvert, 0);
    bindTexture (0, overlay.texture, layer_.uTexture);
    drawQuad();
}

unsigned FrameRenderer::renderComposite (const LayerDesc* layers, int numLayers,
                                         const ImageLayerDesc* overlays, int numOverlays)
{
    lastError_.clear();
    particleBackend_ = "none";
    for (int i = 0; i < numLayers; ++i)
        if (layers[i].shaderOperationPlan != nullptr || layers[i].flatShaderBridge
            || layers[i].curatedShaderTransition != nullptr)
        {
            std::string prepareError;
            if (!prepareFlatShaderBridge(layers[i], prepareError))
            {
                lastError_ = prepareError;
                compositorBackend_ = metalOnly_ ? "metal-rejected" : "opengl-rejected";
                return 0;
            }
            shaderPlanClips_.insert(layers[i].clipId);
        }
    (activeFrameCandidate_ ? activeFrameCandidate_->frameParity : frameParity_)++;   // candidate-local until publication
    // Native score sources begin as deterministic CPU RGBA, then become ordinary
    // backend-owned textures before either compositor sees the layer. This keeps
    // Metal and OpenGL on the same source pixels and lets the existing effects,
    // transform, mask, blend, inspection and export paths remain authoritative.
    std::vector<LayerDesc> preparedLayers;
    bool haveScoreSource = false;
    for (int i = 0; i < numLayers; ++i)
        if (layers[i].scoreSource) { haveScoreSource = true; break; }
    if (haveScoreSource)
    {
        preparedLayers.assign(layers, layers + numLayers);
        for (auto& layer : preparedLayers)
        {
            if (!layer.scoreSource)
                continue;
            if (!canonicalblockc::valid(layer.canonicalBlockCFrame))
            {
                scoreTextures_[layer.clipId] = 0;
                continue;
            }
            const ScoreClock scoreClock { static_cast<float>(layer.shaderClock.beat),
                                          static_cast<float>(layer.shaderClock.beatsPerBar) };
            const auto rendered = renderScore(
                layer.canonicalBlockCFrame, scoreClock, outW_, outH_, layer.genParams);
            auto& texture = scoreTextures_[layer.clipId];
            texture = rendered.rgba.empty() ? 0
                : uploadRgba(rendered.rgba.data(), rendered.width, rendered.height,
                             rendered.width * 4, texture);
            layer.texture = texture;
            layer.texWidth = rendered.width;
            layer.texHeight = rendered.height;
            layer.scoreSource = false;
        }
        layers = preparedLayers.data();
    }
    const auto exactNativeDescriptor = [] (
        const arbitgpu::NativeTextureViewDescriptor& descriptor,
        const char* backend, arbitgpu::NativeTexturePixelFormat format,
        std::uintptr_t view, int width, int height)
    {
        return descriptor.complete() && descriptor.backend == backend
            && descriptor.format == format && descriptor.textureViewHandle == view
            && descriptor.width == static_cast<std::uint32_t>(width)
            && descriptor.height == static_cast<std::uint32_t>(height);
    };
    for (int i = 0; i < numLayers; ++i)
    {
        const auto& layer = layers[i];
        if (layer.nativeTextureBackend == "opengl"
            && (!exactNativeDescriptor(layer.nativeTextureDescriptor, "opengl",
                                       arbitgpu::NativeTexturePixelFormat::Rgba8Unorm,
                                       layer.nativeTextureView, layer.texWidth, layer.texHeight)
                || layer.texture != layer.nativeTextureView || !glIsTexture(layer.texture)))
        {
            lastError_ = "OpenGL native color descriptor is stale or incompatible";
            return 0;
        }
        if ((layer.depthFog || layer.depthEffect != 0)
            && layer.nativeDepthTextureBackend == "opengl"
            && (!exactNativeDescriptor(layer.nativeDepthTextureDescriptor, "opengl",
                                       arbitgpu::NativeTexturePixelFormat::R32Float,
                                       layer.nativeDepthTextureView,
                                       layer.depthWidth, layer.depthHeight)
                || layer.depthTexture != layer.nativeDepthTextureView
                || !glIsTexture(layer.depthTexture)))
        {
            lastError_ = "OpenGL native depth descriptor must name a live exact R32F texture";
            return 0;
        }
    }
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        if (const unsigned nativeTexture = metalRenderer_->renderComposite (
                gl_, layers, numLayers, overlays, numOverlays))
        {
            particleBackend_ = metalRenderer_->particleBackend();
            compositorBackend_ = "metal";
            return nativeTexture;
        }
    if (metalOnly_)
    {
        compositorBackend_ = "metal-rejected";
        return 0;
    }
#endif
    bool requiresMetalTexture = false;
    for (int i = 0; i < numLayers && ! requiresMetalTexture; ++i)
    {
        requiresMetalTexture = layers[i].nativeTextureBackend == "metal";
        if (! requiresMetalTexture && layers[i].fromLayer != nullptr)
            requiresMetalTexture = layers[i].fromLayer->nativeTextureBackend == "metal";
    }
    if (requiresMetalTexture)
    {
        compositorBackend_ = "metal-rejected";
        return 0;
    }
    compositorBackend_ = "opengl";
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    glViewport (0, 0, outW_, outH_);

    // Conscious fixes vs the reference (compositor-spec §6.3): the background
    // clear lands on the first read buffer (quirk 1), fixed-function blending
    // stays OFF — compositing is done entirely in-shader (quirk 2), and every
    // layer is pre-rendered into its own full-frame texture so the blend pass
    // samples back/front at screen UV with identity transform (quirk 3:
    // correct for transformed/cropped front layers).
    glDisable (GL_BLEND);

    int read = 0;
    attachTarget (accumTex_[read]);
    glClearColor (bgR_, bgG_, bgB_, bgA_);   // M1 canvas background param (default charcoal)
    glClear (GL_COLOR_BUFFER_BIT);

    std::vector<bool> overlayComposited ((size_t) std::max (numOverlays, 0), false);
    auto compositeOverlay = [&] (int overlayIndex)
    {
        const ImageLayerDesc& ov = overlays[overlayIndex];
        if (ov.texture == 0 || ov.opacity <= 0.0f)
            return;
        drawImageOverlay (ov, layerTexB_);
        blendOnto (layerTexB_, accumTex_[read], accumTex_[read ^ 1],
                   ov.opacity, 0 /* normal */);
        read ^= 1;
        overlayComposited[(size_t) overlayIndex] = true;
    };

    for (int i = 0; i < numLayers; ++i)
    {
        const LayerDesc& layer = layers[i];
        if (layer.texture == 0 && ! layer.shaderSource && ! layer.flatShaderBridge
            && layer.shaderOperationPlan == nullptr
            && ! layer.particleSource && ! layer.drawShape
            && layer.transitionType == 0 && ! layer.isAdjustment)
            continue;

        // Adjustment layer (M8 "effect the world"): no source of its own — run
        // THIS clip's effect rack over the composite beneath it (accumTex_[read])
        // and mix the result back over that composite at the clip's opacity
        // (1 = full adjustment, <1 = partial). Pointing buildLayerFrame at the
        // accumulator runs the rack and then the clip's transform/crop/MASK
        // (alpha-multiplied), so a masked/transformed adjustment grades only its
        // region (outside the mask front.a == 0 → the composite shows through
        // unchanged); a full-frame clip (identity transform, no mask) reduces to
        // the rack over the whole accumulator. A rack with no enabled effects
        // early-returns the accumulator → an opacity-1 no-op (masked) copy.
        if (layer.isAdjustment)
        {
            // Text created while this adjustment clip was selected belongs to
            // the adjusted composite, not to a detached top-level pass. Insert
            // it immediately before the adjustment so transform/crop/mask,
            // effects, opacity, and their automation process text and video as
            // one image. Project text and text owned by normal clips remain in
            // the ordinary top overlay pass below.
            for (int ov = 0; ov < numOverlays; ++ov)
                if (! overlayComposited[(size_t) ov]
                    && overlays[ov].ownerClipId == layer.clipId)
                    compositeOverlay (ov);

            LayerDesc adj = layer;
            adj.texture        = accumTex_[read];
            adj.texWidth       = outW_;
            adj.texHeight      = outH_;
            adj.transitionType = 0;          // an adjustment is never a transition
            float blendOp = 1.0f;
            const unsigned front = buildLayerFrame (adj, blendOp);
            if (front == 0) return 0;
            blendOnto (front, accumTex_[read], accumTex_[read ^ 1],
                       blendOp, 0 /* normal mix-back */);
            read ^= 1;
            continue;
        }

        // Shader-generator source: synthesise the layer texture into the clip's
        // generator target, then run the normal effects/transform/blend chain
        // on it exactly like a decoded frame. Only the shader path pays a copy.
        const LayerDesc* use = &layer;
        LayerDesc local;
        if (layer.flatShaderBridge && layer.shaderOperationPlan == nullptr)
        {
            const FlatShaderKey key { layer.clipId, layer.visualPlanStructuralRevision, {},
                                      layer.flatShaderNodeId, layer.flatShaderSourceSha256 };
            const auto bridge = flatShaderBridges_.find(key);
            if (bridge == flatShaderBridges_.end() || bridge->second == nullptr)
                continue;
            if (layer.flatShaderFilter && layer.texture == 0)
                continue;
            const unsigned texture = bridge->second->render(
                gl_, layer.shaderClock, outW_, outH_,
                layer.audioPresent ? &layer.audioFeatures : nullptr,
                canonicalblockc::valid(layer.canonicalBlockCFrame)
                    ? layer.canonicalBlockCFrame.get() : nullptr,
                &layer.genParams,
                nullptr, layer.flatShaderFilter ? layer.texture : 0);
            if (texture == 0) continue;
            local = layer;
            local.texture = texture;
            local.texWidth = outW_;
            local.texHeight = outH_;
            use = &local;
        }
        else if (layer.shaderSource)
        {
            const unsigned gtex = renderClipShaderToTexture (
                layer.clipId, layer.shaderClock,
                layer.audioPresent ? &layer.audioFeatures : nullptr,
                canonicalblockc::valid(layer.canonicalBlockCFrame)
                    ? layer.canonicalBlockCFrame.get() : nullptr,
                layer.genParams.empty() ? nullptr : &layer.genParams);
            // The generator bound its own FBO + viewport; restore ours before
            // the layer/effects passes attach to fbo_.
            gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
            glViewport (0, 0, outW_, outH_);
            if (gtex == 0)
                continue;          // no compiled program yet → render nothing
            local = layer;
            if (layer.effects == &layer.graphFeedbackEffect)
                local.effects = &local.graphFeedbackEffect;
            local.texture = gtex;
            local.texWidth = outW_;
            local.texHeight = outH_;
            use = &local;
        }
        else if (layer.particleSource)
        {
            // P4 particle clip: advance + rasterise the clip's compute particle
            // pool into a layer texture, then run the SAME effects/transform/blend
            // chain as a shader/decoded layer. v1 params ride genParams; the
            // spawn-track's notes in the canonical frame seed the pool.
            ParticleParams pp;
            if (layer.particleStateReset)
            {
                const auto existing = particleGens_.find(layer.clipId);
                if (existing != particleGens_.end())
                    existing->second->resetSimulation(gl_);
            }
            auto gp = [&layer] (const char* k, double dflt) -> double {
                const auto it = layer.genParams.find (k);
                return it != layer.genParams.end() ? it->second : dflt;
            };
            pp.count      = (int) (gp ("count", 512.0) + 0.5);
            pp.spawnTrack = (int) (gp ("spawnTrack", 0.0) + 0.5);
            pp.size       = (float) gp ("size", 2.0);
            pp.gravity    = (float) gp ("gravity", 0.0);
            pp.force      = (float) gp ("force", 1.0);
            const bool nativeBuiltin = gp ("nativeBuiltin", 0.0) >= 0.5;
            pp.seed       = (int) gp ("seed", 0.0);
            pp.lifetime   = (float) gp ("lifetime", 0.0);
            if (nativeBuiltin)
            {
                pp.force = (float) gp ("speed", 1.0);
                pp.red = (float) gp ("red", 0.2);
                pp.green = (float) gp ("green", 0.7);
                pp.blue = (float) gp ("blue", 1.0);
                pp.alpha = (float) gp ("alpha", 1.0);
            }
            const canonicalblockc::CanonicalBlockCFrame* particleNotes =
                canonicalblockc::valid(layer.canonicalBlockCFrame)
                    ? layer.canonicalBlockCFrame.get() : nullptr;
            // Procedural trigger particles never fabricate a second score carrier.
            // Without a canonical score frame, the particle engine uses zero feed.
            const unsigned gtex = renderClipParticlesToTexture (
                layer.clipId, layer.visualPlanStructuralRevision, layer.particleNodeId,
                layer.visualPlanTelemetryHold, layer.shaderClock, pp, particleNotes);
            gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
            glViewport (0, 0, outW_, outH_);
            if (gtex == 0)
                continue;          // compute unavailable → render nothing
            local = layer;
            if (layer.effects == &layer.graphFeedbackEffect)
                local.effects = &local.graphFeedbackEffect;
            local.texture = gtex;
            local.texWidth = outW_;
            local.texHeight = outH_;
            use = &local;
        }

        if (use->texture != 0 || use->shaderSource || use->shaderOperationPlan != nullptr
            || use->particleSource
            || use->transitionType != 0)
        {
            float blendOpacity = 1.0f;
            const unsigned front = buildLayerFrame (*use, blendOpacity);
            if (front == 0)
                return 0;
            if (use->clipId == transformInspectionClip_ && use->transitionType == 0
                && ! use->inspectionDrawShapeOutput)
            {
                if (transformInspectionTex_ == 0)
                    transformInspectionTex_ = createOutputTexture();
                attachTarget (transformInspectionTex_);
                gl_->UseProgram (blit_.prog);
                bindTexture (0, front, blit_.uTexture);
                drawQuad();
                transformInspectionReady_ = true;
            }
            // buildLayerFrame rebinds attachments; restore viewport-sized target.
            blendOnto (front, accumTex_[read], accumTex_[read ^ 1],
                       blendOpacity, use->blendMode);
            read ^= 1;
        }

        if (use->drawShape)
        {
            const auto drawShapeStarted = std::chrono::steady_clock::now();
            attachTarget (layerTexB_);
            glClearColor (0.0f, 0.0f, 0.0f, 0.0f);
            glClear (GL_COLOR_BUFFER_BIT);
            gl_->UseProgram (drawShape_.prog);
            setPassthrough (drawShape_.uTransform, drawShape_.uCrop);
            gl_->Uniform4f (drawShape_.uRect, use->drawShapeCx, use->drawShapeCy,
                             use->drawShapeEllipse ? -std::max (use->drawShapeW, 0.0f)
                                                   : std::max (use->drawShapeW, 0.0f),
                             std::max (use->drawShapeH, 0.0f));
            gl_->Uniform4f (drawShape_.uRectB, use->drawShape2Cx, use->drawShape2Cy,
                            use->drawShapeSecondaryEllipse ? -std::max (use->drawShape2W, 0.0f)
                                                           : std::max (use->drawShape2W, 0.0f),
                            std::max (use->drawShape2H, 0.0f));
            gl_->Uniform1i (drawShape_.uOperation,
                            use->drawShapeHasSecondary
                                ? std::clamp (use->drawShapeOperation, 1, 3) : 0);
            gl_->Uniform4f (drawShape_.uColor,
                            std::clamp (use->drawShapeR, 0.0f, 1.0f),
                            std::clamp (use->drawShapeG, 0.0f, 1.0f),
                            std::clamp (use->drawShapeB, 0.0f, 1.0f),
                            std::clamp (use->drawShapeA, 0.0f, 1.0f));
            drawQuad();
            if (visualTelemetry_ != nullptr && use->drawShapeNodeId != 0)
                visualTelemetry_->recordNodeEvaluation(use->clipId,
                    use->visualPlanStructuralRevision, use->drawShapeNodeId,
                    static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - drawShapeStarted).count()),
                    use->visualPlanTelemetryHold);
            if (use->inspectionDrawShapeOutput
                && use->clipId == transformInspectionClip_)
            {
                if (transformInspectionTex_ == 0)
                    transformInspectionTex_ = createOutputTexture();
                attachTarget (transformInspectionTex_);
                gl_->UseProgram (blit_.prog);
                bindTexture (0, layerTexB_, blit_.uTexture);
                drawQuad();
                transformInspectionReady_ = true;
            }
            blendOnto (layerTexB_, accumTex_[read], accumTex_[read ^ 1], 1.0f, 0);
            read ^= 1;
        }
    }

    for (int i = 0; i < numOverlays; ++i)
    {
        if (! overlayComposited[(size_t) i])
            compositeOverlay (i);
    }

    // HDR post stack (bloom + tonemap) over the final composite. Neutral by
    // default ⇒ returns accumTex_[read] unchanged (and leaves FBO 0 bound), so
    // the export/viewport readback path is untouched until a caller opts in via
    // setPostFx. preview==export by construction: both paths call renderComposite.
    return applyPostFx (accumTex_[read]);
}

#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
bool FrameRenderer::renderCompositeToIOSurface (
    void* ioSurface, int width, int height,
    const LayerDesc* layers, int numLayers,
    const ImageLayerDesc* overlays, int numOverlays)
{
    particleBackend_ = "none";
    for (int i = 0; i < numLayers; ++i)
        if (layers[i].shaderOperationPlan != nullptr || layers[i].flatShaderBridge
            || layers[i].curatedShaderTransition != nullptr)
        {
            std::string prepareError;
            if (!prepareFlatShaderBridge(layers[i], prepareError))
            {
                lastError_ = prepareError;
                compositorBackend_ = "metal-zero-copy-rejected";
                return false;
            }
            shaderPlanClips_.insert(layers[i].clipId);
        }
    (activeFrameCandidate_ ? activeFrameCandidate_->frameParity : frameParity_)++;
    if (metalRenderer_ == nullptr || ioSurface == nullptr)
    {
        lastError_ = metalRenderer_ == nullptr
            ? "Metal renderer is unavailable"
            : "Metal IOSurface is unavailable";
        compositorBackend_ = "metal-zero-copy-unavailable";
        return false;
    }
    const bool rendered = metalRenderer_->renderCompositeToIOSurface (
        gl_, ioSurface, width, height, layers, numLayers, overlays, numOverlays);
    lastError_ = rendered ? std::string() : metalRenderer_->lastError();
    particleBackend_ = metalRenderer_->particleBackend();
    compositorBackend_ = rendered ? "metal-zero-copy" : "metal-zero-copy-rejected";
    return rendered;
}

void FrameRenderer::clearMetalDirectOutputs()
{
    if (metalRenderer_ != nullptr) metalRenderer_->clearDirectOutputs();
}

void FrameRenderer::setMetalInspectionPresentation (
    unsigned requestedHandle, const videopreview::State& state)
{
    metalInspectionRequestedHandle_ = requestedHandle;
    metalInspectionPresentation_ = state;
    if (metalRenderer_ != nullptr)
        metalRenderer_->setInspection (transformInspectionClip_, requestedHandle, state);
}

bool FrameRenderer::metalInspectionResourceAdmitted() const
{
    return metalOnly_ && metalRenderer_ != nullptr
        && metalRenderer_->inspectionResourceAdmitted();
}
#endif

// ---- Shader-generator sources (M3) ---------------------------------------

bool FrameRenderer::prepareFlatShaderBridge (const LayerDesc& layer, std::string& error)
{
    if (layer.shaderOperationPlan == nullptr)
    {
        const bool transition = layer.curatedShaderTransition != nullptr;
        if (!layer.flatShaderBridge && !transition) return true;
        if (gl_ == nullptr)
        {
            error = transition
                ? "curated shader transition requires the OpenGL shader execution seam"
                : "FlatShaderBridge requires the OpenGL shader execution seam";
            return false;
        }
        const auto* operation = transition ? layer.curatedShaderTransition.get() : nullptr;
        if (transition)
        {
            std::string contractError;
            std::map<std::string, double> parameters;
            const auto* catalog = shadercatalog::find(operation->payload.catalogPackId,
                                                       operation->payload.catalogProgramId);
            if (!shadertransition::validate(operation->payload, contractError)
                || videohelper::sha256Text(operation->payload.source)
                       != operation->payload.sourceSha256
                || catalog == nullptr
                || !shadercatalog::validateParameterWire(*catalog, operation->payload.parameters,
                                                          parameters, contractError)
                || parameters != operation->parameterValues)
            {
                error = contractError.empty()
                    ? "curated shader transition immutable operation is stale or malformed"
                    : contractError;
                return false;
            }
        }
        else
        {
            error = "FlatShaderBridge requires an immutable shader operation plan";
            return false;
        }
        const FlatShaderKey key { layer.clipId, layer.visualPlanStructuralRevision, {},
                                  transition ? operation->nodeId : layer.flatShaderNodeId,
                                  transition ? operation->payload.sourceSha256
                                             : layer.flatShaderSourceSha256 };
        if (flatShaderBridges_.find(key) != flatShaderBridges_.end()) return true;
        auto candidate = std::make_unique<ShaderGenerator>();
        const auto dialect = transition || layer.flatShaderIsf
            ? arbitshader::Dialect::Isf : arbitshader::Dialect::BareGlsl;
        const std::string& source = transition ? operation->payload.source : layer.flatShaderSource;
        if (!candidate->setBridgeSource(gl_, source, dialect,
                                        !transition && layer.flatShaderFilter,
                                        nullptr))
        {
            error = candidate->log();
            candidate->shutdown(gl_);
            return false;
        }
        flatShaderBridges_.emplace(key, std::move(candidate));
        error.clear();
        return true;
    }
    const auto plan = layer.shaderOperationPlan;
    if (plan == nullptr || plan->operations.empty()
        || plan->operations.size() > videowire::ShaderOperationPlan::maximumOperations
        || plan->passTargetCount > videowire::ShaderOperationPlan::maximumPassTargets)
    {
        error = "shader operation plan is missing or exceeds its resource budget";
        return false;
    }
    if (plan->revision != layer.visualPlanStructuralRevision
        || plan->digest.empty()
        || plan->digest != videowire::shaderOperationPlanDigest(*plan, plan->revision))
    { error = "shader operation plan identity differs from its immutable content"; return false; }
    std::size_t countedPasses = 0;
    int priorOutput = 0;
    std::set<int> operationNodes;
    std::set<int> outputResources;
    for (const auto& operation : plan->operations)
    {
        countedPasses += operation.payload.passResources.passes.size();
        videowire::CuratedIsfPassResources sourcePasses;
        if (operation.payload.language == videowire::FlatShaderLanguage::isf
            && (!videowire::admitCuratedIsfPassResources(
                    operation.payload.source, sourcePasses, error)
                || !(sourcePasses == operation.payload.passResources)))
        {
            if (error.empty()) error = "shader operation multipass schedule differs from its source";
            return false;
        }
        if (operation.nodeId == 0 || operation.outputNodeId == 0
            || videohelper::sha256Text(operation.payload.source) != operation.payload.sourceSha256
            || operation.generatedParameters != operation.payload.parameterValues
            || countedPasses > videowire::ShaderOperationPlan::maximumPassTargets
            || !videowire::admitCuratedIsfPassExtent(
                operation.payload.passResources, outW_, outH_, error))
        { if (error.empty()) error = "shader operation plan lost its exact admitted payload"; return false; }
        if (operation.kind == videowire::ShaderOperationKind::generator)
        {
            if (operation.inputCount != 0) { error = "shader generator has an input resource"; return false; }
        }
        else if (operation.kind == videowire::ShaderOperationKind::filter)
        {
            if (operation.inputCount != 1 || operation.inputNodeIds[0] == 0
                || (priorOutput != 0 && operation.inputNodeIds[0] != priorOutput))
            { error = "shader filter input resource identity is not the prior output"; return false; }
        }
        else if (!operation.transitionPayload || operation.inputCount != 2
                 || operation.inputNodeIds[0] == 0 || operation.inputNodeIds[1] == 0
                 || operation.inputNodeIds[0] == operation.inputNodeIds[1]
                 || (priorOutput != 0 && operation.inputNodeIds[0] != priorOutput
                     && operation.inputNodeIds[1] != priorOutput))
        { error = "shader transition requires two exact ordered input resources"; return false; }
        if (operation.customGrant)
        {
            const auto kind = operation.payload.language == videowire::FlatShaderLanguage::isf
                ? programmableruntime::PayloadKind::isf : programmableruntime::PayloadKind::shader;
            if (!programmableruntime::admits(*operation.customGrant, kind,
                    operation.payload.source, error)
                || operation.customGrant->verifiedBundledCurated
                || !operation.customGrant->catalogPackId.empty()
                || !operation.customGrant->catalogProgramId.empty()
                || !operation.payload.catalogPackId.empty()
                || !operation.payload.catalogProgramId.empty())
            { if (error.empty()) error = "custom shader grant no longer owns the exact immutable payload"; return false; }
        }
        else
        {
            const auto* catalog = shadercatalog::find(operation.payload.catalogPackId,
                                                      operation.payload.catalogProgramId);
            if (catalog == nullptr)
            { error = "shader operation manifest authority is unavailable"; return false; }
            if (catalog->sourceSha256 != operation.payload.sourceSha256)
            { error = "shader operation source differs from catalog authority"; return false; }
        }
        if (!operationNodes.insert(operation.nodeId).second
            || !outputResources.insert(operation.outputNodeId).second)
        { error = "shader operation resource identity is repeated"; return false; }
        priorOutput = operation.nodeId;
    }
    if (countedPasses != plan->passTargetCount)
    { error = "shader operation plan resource budget is stale"; return false; }
    countedPasses = 0;
    bool useMetal = false;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    useMetal = metalOnly_;
#endif
    std::map<FlatShaderKey, std::unique_ptr<ShaderGenerator>> staged;
    struct StagedShaderCleanup
    {
        std::map<FlatShaderKey, std::unique_ptr<ShaderGenerator>>& shaders;
        const arbitgl::GlFuncs* gl = nullptr;
        bool committed = false;
        ~StagedShaderCleanup()
        {
            if (committed || gl == nullptr) return;
            for (auto& shader : shaders)
                if (shader.second != nullptr) shader.second->shutdown(gl);
        }
    } stagedCleanup { staged, gl_ };
    priorOutput = 0;
    for (const auto& operation : plan->operations)
    {
        if (operation.nodeId == 0 || operation.outputNodeId == 0
            || videohelper::sha256Text(operation.payload.source) != operation.payload.sourceSha256
            || operation.generatedParameters != operation.payload.parameterValues)
        { error = "shader operation plan lost its exact admitted payload"; return false; }
        countedPasses += operation.payload.passResources.passes.size();
        if (operation.kind == videowire::ShaderOperationKind::generator)
        {
            if (operation.inputCount != 0) { error = "shader generator has an input resource"; return false; }
        }
        else if (operation.kind == videowire::ShaderOperationKind::filter)
        {
            if (operation.inputCount != 1 || operation.inputNodeIds[0] == 0
                || (priorOutput != 0 && operation.inputNodeIds[0] != priorOutput))
            { error = "shader filter input resource identity is not the prior output"; return false; }
        }
        else if (operation.inputCount != 2 || operation.inputNodeIds[0] == 0
                 || operation.inputNodeIds[1] == 0
                 || operation.inputNodeIds[0] == operation.inputNodeIds[1])
        { error = "shader transition requires exactly two input resources"; return false; }
        if (!videowire::admitCuratedIsfPassExtent(operation.payload.passResources, outW_, outH_, error))
            return false;
        const FlatShaderKey key { layer.clipId, layer.visualPlanStructuralRevision,
                                  plan->digest,
                                  operation.nodeId, operation.payload.sourceSha256 };
        if (!useMetal && flatShaderBridges_.find(key) == flatShaderBridges_.end())
        {
            auto candidate = std::make_unique<ShaderGenerator>();
            const auto dialect = operation.payload.language == videowire::FlatShaderLanguage::isf
                ? arbitshader::Dialect::Isf : arbitshader::Dialect::BareGlsl;
            if (!candidate->setBridgeSource(gl_, operation.payload.source, dialect,
                    operation.kind == videowire::ShaderOperationKind::filter,
                    operation.payload.language == videowire::FlatShaderLanguage::isf
                        ? &operation.payload.passResources : nullptr))
            { error = candidate->log(); candidate->shutdown(gl_); return false; }
            const auto* catalog = shadercatalog::find(operation.payload.catalogPackId,
                                                       operation.payload.catalogProgramId);
            std::vector<videowire::FlatShaderRuntimeParameter> runtimeParameters;
            for (const auto& parameter : candidate->params())
            {
                const auto components = genParamComponentCount(parameter.type);
                if (components == 0) continue;
                const char* type = parameter.type == 0 ? "bool" : parameter.type == 1 ? "long"
                    : parameter.type == 2 ? "float" : parameter.type == 3 ? "point2D"
                    : parameter.type == 4 ? "color" : "";
                runtimeParameters.push_back({parameter.name, type, static_cast<std::size_t>(components)});
            }
            const bool valid = operation.customGrant.has_value()
                ? runtimeParameters.empty() && operation.generatedParameters.empty()
                : catalog != nullptr && videowire::validateFlatShaderRuntimeParameters(
                    *catalog, runtimeParameters, operation.generatedParameters, error);
            if (!valid)
            { if (error.empty()) error = "shader operation parameters differ from the generated manifest";
              candidate->shutdown(gl_); return false; }
            staged.emplace(key, std::move(candidate));
        }
        priorOutput = operation.nodeId;
    }
    if (countedPasses != plan->passTargetCount)
    { error = "shader operation plan resource budget is stale"; return false; }
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_)
    {
        if (metalRenderer_ == nullptr)
        { error = "shader operation plan requires the native Metal backend"; return false; }
        return metalRenderer_->prepareShaderOperationPlan(layer, error);
    }
#endif
    if (gl_ == nullptr)
    { error = "shader operation plan requires the OpenGL execution backend"; return false; }
    for (auto it = flatShaderBridges_.begin(); it != flatShaderBridges_.end();)
    {
        if (it->first.clipId == layer.clipId
            && (it->first.revision != layer.visualPlanStructuralRevision
                || it->first.planDigest != plan->digest))
        {
            if (it->second != nullptr) it->second->shutdown(gl_);
            it = flatShaderBridges_.erase(it);
        }
        else ++it;
    }
    for (auto& candidate : staged)
        flatShaderBridges_.emplace(candidate.first, std::move(candidate.second));
    stagedCleanup.committed = true;
    error.clear();
    return true;
}

bool FrameRenderer::setClipShader (int clipId, const std::string& source,
                                   std::string& logOut, std::vector<GenParam>& paramsOut)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_ && metalRenderer_ != nullptr)
        return metalRenderer_->setClipShader (clipId, source, logOut, paramsOut);
#endif
    if (gl_ == nullptr) { logOut = "renderer not initialized"; return false; }
    auto& gen = shaderGens_[clipId];
    if (gen == nullptr)
        gen = std::make_unique<ShaderGenerator>();
    const bool ok = gen->setSource (gl_, source);
    logOut = gen->log();
    paramsOut = gen->params();
    return ok;
}

void FrameRenderer::clearClipShader (int clipId)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_ && metalRenderer_ != nullptr)
        metalRenderer_->clearClipShader (clipId);
#endif
    auto it = shaderGens_.find (clipId);
    if (it != shaderGens_.end())
    {
        if (it->second != nullptr && gl_ != nullptr)
            it->second->shutdown (gl_);
        shaderGens_.erase (it);
    }
    for (auto bridge = flatShaderBridges_.begin(); bridge != flatShaderBridges_.end();)
    {
        if (bridge->first.clipId == clipId)
        {
            if (bridge->second != nullptr && gl_ != nullptr) bridge->second->shutdown(gl_);
            bridge = flatShaderBridges_.erase(bridge);
        }
        else ++bridge;
    }
    // A clip's decoded image-INPUT textures (M7 image follow-up) live and die
    // with its shader program.
    auto im = clipImageTex_.find (clipId);
    if (im != clipImageTex_.end())
    {
        if (gl_ != nullptr)
            for (auto& [name, tex] : im->second)
                if (tex != 0) deleteTexture (tex);
        clipImageTex_.erase (im);
    }
    shaderPlanClips_.erase(clipId);
}

void FrameRenderer::retainShaderPlanClips (const std::set<int>& clipIds)
{
    const auto prior = shaderPlanClips_;
    for (const int clipId : prior)
        if (clipIds.count(clipId) == 0)
            clearClipShader(clipId);
    shaderPlanClips_ = clipIds;
}

bool FrameRenderer::hasPreparedShaderPlan (int clipId) const
{
    return std::any_of(flatShaderBridges_.begin(), flatShaderBridges_.end(),
        [clipId](const auto& bridge) { return bridge.first.clipId == clipId; });
}

void FrameRenderer::setClipImage (int clipId, const std::string& name,
                                  const uint8_t* rgba, int width, int height, int strideBytes)
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_ && metalRenderer_ != nullptr)
    {
        metalRenderer_->setClipImage (clipId, name, rgba, width, height, strideBytes);
        return;
    }
#endif
    if (gl_ == nullptr || rgba == nullptr || width <= 0 || height <= 0)
        return;
    auto& byName = clipImageTex_[clipId];
    unsigned existing = 0;
    if (const auto it = byName.find (name); it != byName.end())
        existing = it->second;
    byName[name] = uploadRgba (rgba, width, height, strideBytes, existing);
}

bool FrameRenderer::hasClipShader (int clipId) const
{
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalOnly_ && metalRenderer_ != nullptr)
        return metalRenderer_->hasClipShader (clipId);
#endif
    auto it = shaderGens_.find (clipId);
    return it != shaderGens_.end() && it->second != nullptr && it->second->hasProgram();
}

unsigned FrameRenderer::renderClipShaderToTexture (int clipId, const ShaderClock& clock,
                                                   const AudioFeatures* audio,
                                                   const canonicalblockc::CanonicalBlockCFrame* notes,
                                                   const std::map<std::string, double>* genValues)
{
    auto it = shaderGens_.find (clipId);
    if (it == shaderGens_.end() || it->second == nullptr)
        return 0;
    const std::map<std::string, unsigned>* imgs = nullptr;
    if (const auto im = clipImageTex_.find (clipId);
        im != clipImageTex_.end() && ! im->second.empty())
        imgs = &im->second;
    return videowire::runVisualRendererTelemetryOperation(
        visualTelemetry_, { videowire::VisualExecutionKind::generator },
        [&] { return it->second->render(gl_, clock, outW_, outH_, audio, notes,
                                       genValues, imgs); });
}

unsigned FrameRenderer::renderClipParticlesToTexture (int clipId, uint64_t structuralRevision,
                                                      int stableNodeId, bool telemetryHold,
                                                      const ShaderClock& clock,
                                                      const ParticleParams& params,
                                                      const canonicalblockc::CanonicalBlockCFrame* notes)
{
    if (gl_ == nullptr) return 0;
    auto& eng = particleGens_[clipId];
    if (eng == nullptr)
        eng = std::make_unique<ParticleEngine>();
    const unsigned texture = videowire::runVisualRendererTelemetryOperation(
        visualTelemetry_, { videowire::VisualExecutionKind::particle, clipId,
                            structuralRevision, stableNodeId, telemetryHold },
        [&] { return eng->render(gl_, clock, outW_, outH_, params, notes); });
    particleBackend_ = eng->log().empty() ? "pending" : eng->log();
    return texture;
}

void FrameRenderer::setPostFx (float bloomIntensity, float bloomThreshold,
                              float bloomRadius, int tonemap, float exposure)
{
    bloomIntensity_ = bloomIntensity > 0.0f ? bloomIntensity : 0.0f;
    bloomThreshold_ = bloomThreshold;
    bloomRadius_    = bloomRadius > 0.0f ? bloomRadius : 0.0f;
    tonemapMode_    = (tonemap >= 0 && tonemap <= 2) ? tonemap : 0;
    exposure_       = exposure > 0.0f ? exposure : 1.0f;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setPostFx (bloomIntensity_, bloomThreshold_, bloomRadius_,
                                   tonemapMode_, exposure_);
#endif
}

void FrameRenderer::setBackgroundColor (float r, float g, float b, float a)
{
    bgR_ = r; bgG_ = g; bgB_ = b; bgA_ = a;
#if defined(__APPLE__) && ARBIT_HAVE_METAL_BACKEND
    if (metalRenderer_ != nullptr)
        metalRenderer_->setBackgroundColor (r, g, b, a);
#endif
}

unsigned FrameRenderer::applyPostFx (unsigned compositeTexture)
{
    const bool bloomOn   = bloomIntensity_ > 0.0f && bloomRadius_ > 0.0f
                        && bloomThresh_.prog != 0 && blur_.prog != 0;
    const bool tonemapOn = tonemapMode_ != 0 || exposure_ != 1.0f;

    // Neutral ⇒ skip entirely: return the composite untouched (byte-identical to
    // a no-post job). Leave FBO 0 bound, matching renderComposite's old tail.
    if (gl_ == nullptr || bloomCombine_.prog == 0 || (! bloomOn && ! tonemapOn))
    {
        gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
        return compositeTexture;
    }

    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    glViewport (0, 0, outW_, outH_);
    glDisable (GL_BLEND);

    if (bloomOn)
    {
        // 1. bright-pass: composite → fxTex_[0] (free after the composite loop).
        attachTarget (fxTex_[0]);
        gl_->UseProgram (bloomThresh_.prog);
        setPassthrough (bloomThresh_.uTransform, bloomThresh_.uCrop);
        bindTexture (0, compositeTexture, bloomThresh_.uTexture);
        gl_->Uniform1f (bloomThresh_.uThreshold, bloomThreshold_);
        drawQuad();

        // 2. separable gaussian blur, ping-ponging fxTex_[0] <-> fxTex_[1]. The
        // blur shader caps a single pass at radius 20, so a wider bloom spreads
        // over several iterations (one pair widens the effective sigma).
        const float radius = std::min (bloomRadius_, 20.0f);
        const int   passes = std::max (1, std::min (5, (int) std::ceil (bloomRadius_ / 10.0f)));
        gl_->UseProgram (blur_.prog);
        setPassthrough (blur_.uTransform, blur_.uCrop);
        gl_->Uniform1f (blur_.uRadius, radius);
        for (int p = 0; p < passes; ++p)
        {
            attachTarget (fxTex_[1]);                       // horizontal
            bindTexture (0, fxTex_[0], blur_.uTexture);
            gl_->Uniform2f (blur_.uTexelStep, 1.0f / (float) outW_, 0.0f);
            drawQuad();
            attachTarget (fxTex_[0]);                       // vertical
            bindTexture (0, fxTex_[1], blur_.uTexture);
            gl_->Uniform2f (blur_.uTexelStep, 0.0f, 1.0f / (float) outH_);
            drawQuad();
        }
    }

    // 3. combine (+ bloom) + tonemap → the free accum target (the one that does
    // not hold the composite — same trick as applyCanvasFrame).
    const unsigned target = (compositeTexture == accumTex_[0]) ? accumTex_[1] : accumTex_[0];
    attachTarget (target);
    gl_->UseProgram (bloomCombine_.prog);
    setPassthrough (bloomCombine_.uTransform, bloomCombine_.uCrop);
    bindTexture (0, compositeTexture, bloomCombine_.uTexture);
    bindTexture (1, bloomOn ? fxTex_[0] : compositeTexture, bloomCombine_.uBloomTex);
    gl_->Uniform1f (bloomCombine_.uIntensity, bloomOn ? bloomIntensity_ : 0.0f);
    gl_->Uniform1f (bloomCombine_.uExposure, exposure_);
    gl_->Uniform1i (bloomCombine_.uTonemap, tonemapMode_);
    drawQuad();

    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return target;
}

unsigned FrameRenderer::applyCanvasFrame (unsigned compositeTexture)
{
    if (gl_ == nullptr || canvasW_ <= 0 || canvasH_ <= 0 || canvas_.prog == 0)
        return compositeTexture;

    // Output into the present target at the display resolution when one is set
    // (audit #16: the composite ran at the canvas resolution, so we can't reuse
    // a canvas-res accum target for a window-res display). Otherwise frame into
    // the spare accum target at the composite resolution (legacy behavior).
    const bool toPresent = (presentW_ > 0 && presentTex_ != 0);
    const unsigned target = toPresent
        ? presentTex_
        : (compositeTexture == accumTex_[0] ? accumTex_[1] : accumTex_[0]);
    const int vpW = toPresent ? presentW_ : outW_;
    const int vpH = toPresent ? presentH_ : outH_;
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    glViewport (0, 0, vpW, vpH);
    glDisable (GL_BLEND);
    attachTarget (target);

    // Canvas rect in composite UV through the view transform: canvas NDC
    // corners (±1, ±1) map to pan ± zoom·fit.
    float zx, zy, cx, cy;
    viewMap (zx, zy, cx, cy);
    const float minU = (cx - zx + 1.0f) * 0.5f;
    const float maxU = (cx + zx + 1.0f) * 0.5f;
    const float minV = (cy - zy + 1.0f) * 0.5f;
    const float maxV = (cy + zy + 1.0f) * 0.5f;

    gl_->UseProgram (canvas_.prog);
    setPassthrough (canvas_.uTransform, canvas_.uCrop);
    bindTexture (0, compositeTexture, canvas_.uTexture);
    gl_->Uniform4f (canvas_.uCanvasRect, minU, minV, maxU, maxV);
    gl_->Uniform2f (canvas_.uTexel, 1.0f / (float) vpW, 1.0f / (float) vpH);
    drawQuad();
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return target;
}

unsigned FrameRenderer::applyNodePreview (unsigned finalTexture, unsigned previewTexture,
                                           const videopreview::State& state)
{
    if (gl_ == nullptr || state.layout == videopreview::Layout::hidden
        || previewTexture == 0 || preview_.prog == 0)
        return finalTexture;
    const bool toPresent = presentW_ > 0 && presentTex_ != 0;
    const unsigned target = toPresent ? presentTex_
        : (finalTexture == accumTex_[0] ? accumTex_[1] : accumTex_[0]);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (target);
    glViewport (0, 0, toPresent ? presentW_ : outW_, toPresent ? presentH_ : outH_);
    glDisable (GL_BLEND);
    gl_->UseProgram (preview_.prog);
    bindTexture (0, finalTexture, preview_.uFinal);
    bindTexture (1, previewTexture, preview_.uPreview);
    gl_->Uniform1i (preview_.uLayout, static_cast<int> (state.layout));
    gl_->Uniform1i (preview_.uBackground, static_cast<int> (state.background));
    gl_->Uniform1f (preview_.uZoom, state.zoom);
    gl_->Uniform2f (preview_.uPan, state.panX, state.panY);
    gl_->Uniform1f (preview_.uSplit, state.split);
    drawQuad();
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return target;
}

bool FrameRenderer::presentToWindow (unsigned compositeTexture, int fbWidth, int fbHeight)
{
    if (compositeTexture == 0 || fbWidth <= 0 || fbHeight <= 0) return false;
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    glViewport (0, 0, fbWidth, fbHeight);
    glDisable (GL_BLEND);
    gl_->UseProgram (blit_.prog);
    bindTexture (0, compositeTexture, blit_.uTexture);
    drawQuad();
    return glGetError() == GL_NO_ERROR;
}

bool FrameRenderer::renderToPixels (const LayerDesc* layers, int numLayers,
                                    std::vector<uint8_t>& rgbaOut, std::string& error,
                                    const ImageLayerDesc* overlays, int numOverlays)
{
    if (gl_ == nullptr)
    {
        error = "renderer not initialized";
        return false;
    }
    const unsigned tex = renderComposite (layers, numLayers, overlays, numOverlays);
    if (tex == 0)
    {
        rgbaOut.clear();
        error = lastError_.empty() ? "renderer produced no composite texture" : lastError_;
        return false;
    }

    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (tex);
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    rgbaOut.resize ((size_t) outW_ * (size_t) outH_ * 4);
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    // Composite space keeps row 0 = image top, so this is already
    // top-row-first RGBA (DecodedFrame convention).
    glReadPixels (0, 0, outW_, outH_, GL_RGBA, GL_UNSIGNED_BYTE, rgbaOut.data());
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    return true;
}

bool FrameRenderer::renderToPixelsAsync (const LayerDesc* layers, int numLayers,
                                         std::vector<uint8_t>& prevFrameOut, bool& havePrevOut,
                                         std::string& error,
                                         const ImageLayerDesc* overlays, int numOverlays)
{
    havePrevOut = false;
    if (gl_ == nullptr)
    {
        error = "renderer not initialized";
        return false;
    }
    const unsigned tex = renderComposite (layers, numLayers, overlays, numOverlays);
    if (tex == 0)
    {
        prevFrameOut.clear();
        error = lastError_.empty() ? "renderer produced no composite texture" : lastError_;
        return false;
    }

    const int frameBytes = outW_ * outH_ * 4;
    if (exportPbo_[0] == 0 || exportPboBytes_ != frameBytes)
    {
        if (exportPbo_[0] != 0)
            gl_->DeleteBuffers (2, exportPbo_);
        gl_->GenBuffers (2, exportPbo_);
        for (int i = 0; i < 2; ++i)
        {
            gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, exportPbo_[i]);
            gl_->BufferData (GL_PIXEL_PACK_BUFFER, frameBytes, nullptr, GL_STREAM_READ);
        }
        exportPboBytes_ = frameBytes;
        exportPending_[0] = exportPending_[1] = false;
        exportIdx_ = 0;
    }

    // Map the previous call's readback (the GPU has had a full frame's worth
    // of encode time to finish it, so this rarely stalls).
    const int prev = exportIdx_ ^ 1;
    if (exportPending_[prev])
    {
        exportPending_[prev] = false;
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, exportPbo_[prev]);
        if (const void* p = gl_->MapBufferRange (GL_PIXEL_PACK_BUFFER, 0,
                                                 exportPboBytes_, GL_MAP_READ_BIT))
        {
            prevFrameOut.assign (static_cast<const uint8_t*> (p),
                                 static_cast<const uint8_t*> (p) + exportPboBytes_);
            havePrevOut = true;
            gl_->UnmapBuffer (GL_PIXEL_PACK_BUFFER);
        }
        if (! havePrevOut)
        {
            gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
            error = "PBO map failed during export readback";
            return false;
        }
    }

    // Issue this frame's async readback into the other PBO.
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (tex);
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, exportPbo_[exportIdx_]);
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, outW_, outH_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    exportPending_[exportIdx_] = true;
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    exportIdx_ = prev;
    return true;
}

bool FrameRenderer::drainAsyncReadback (std::vector<uint8_t>& rgbaOut)
{
    if (gl_ == nullptr)
        return false;
    for (int i = 0; i < 2; ++i)
    {
        if (! exportPending_[i])
            continue;
        exportPending_[i] = false;
        bool have = false;
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, exportPbo_[i]);
        if (const void* p = gl_->MapBufferRange (GL_PIXEL_PACK_BUFFER, 0,
                                                 exportPboBytes_, GL_MAP_READ_BIT))
        {
            rgbaOut.assign (static_cast<const uint8_t*> (p),
                            static_cast<const uint8_t*> (p) + exportPboBytes_);
            have = true;
            gl_->UnmapBuffer (GL_PIXEL_PACK_BUFFER);
        }
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
        return have;
    }
    return false;
}

bool FrameRenderer::hashComposite (unsigned compositeTexture, uint64_t& hashOut)
{
    if (gl_ == nullptr || hashPbo_[0] == 0 || compositeTexture == 0)
        return false;

    bool have = false;
    const int prev = hashIdx_ ^ 1;
    if (hashPending_[prev])
    {
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, hashPbo_[prev]);
        if (const void* p = gl_->MapBufferRange (GL_PIXEL_PACK_BUFFER, 0,
                                                 hashBytes_, GL_MAP_READ_BIT))
        {
            hashOut = fnv1a (static_cast<const uint8_t*> (p), (size_t) hashBytes_);
            have = true;
            gl_->UnmapBuffer (GL_PIXEL_PACK_BUFFER);
        }
        hashPending_[prev] = false;
    }

    // Issue the next async readback of the composited frame.
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (compositeTexture);
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, hashPbo_[hashIdx_]);
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, outW_, hashRows_, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    hashPending_[hashIdx_] = true;
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    hashIdx_ = prev;
    return have;
}

const uint8_t* FrameRenderer::beginMappedReadback (unsigned texture, int& widthOut, int& heightOut)
{
    // Read at the present (display) resolution when a present target is active —
    // `texture` is the framed presentTex_ (window-sized), NOT a canvas-res accum
    // target (audit #16). Falls back to the composite size for the legacy path.
    const int rbW = presentW_ > 0 ? presentW_ : outW_;
    const int rbH = presentW_ > 0 ? presentH_ : outH_;
    if (gl_ == nullptr || fbo_ == 0 || texture == 0
        || rbW <= 0 || rbH <= 0 || frameMapped_)
        return nullptr;
    if (framePbo_[0] == 0)
        gl_->GenBuffers (2, framePbo_);

    // Map the previous call's readback. The mapping stays valid after the
    // buffer is unbound — only Map/Unmap need the binding.
    const int prev = frameIdx_ ^ 1;
    const uint8_t* mapped = nullptr;
    if (framePending_[prev])
    {
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, framePbo_[prev]);
        mapped = static_cast<const uint8_t*> (gl_->MapBufferRange (
            GL_PIXEL_PACK_BUFFER, 0,
            (GLsizeiptr) framePboW_[prev] * framePboH_[prev] * 4, GL_MAP_READ_BIT));
        if (mapped != nullptr)
        {
            widthOut = framePboW_[prev];
            heightOut = framePboH_[prev];
            frameMapped_ = true;
        }
        framePending_[prev] = false;
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    }

    // Issue this frame's async readback. BGRA is core since GL 1.2 and is
    // the natural framebuffer order on desktop GPUs (no driver swizzle).
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (texture);
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, framePbo_[frameIdx_]);
    if (framePboW_[frameIdx_] != rbW || framePboH_[frameIdx_] != rbH)
    {
        gl_->BufferData (GL_PIXEL_PACK_BUFFER, (GLsizeiptr) rbW * rbH * 4,
                         nullptr, GL_STREAM_READ);
        framePboW_[frameIdx_] = rbW;
        framePboH_[frameIdx_] = rbH;
    }
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, rbW, rbH, GL_BGRA, GL_UNSIGNED_BYTE, nullptr);
    framePending_[frameIdx_] = true;
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    frameIdx_ = prev;
    return mapped;
}

void FrameRenderer::endMappedReadback()
{
    if (! frameMapped_ || gl_ == nullptr)
        return;
    // After the index swap in begin(), the mapped buffer is framePbo_[frameIdx_].
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, framePbo_[frameIdx_]);
    gl_->UnmapBuffer (GL_PIXEL_PACK_BUFFER);
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    frameMapped_ = false;
}

bool FrameRenderer::readbackScope (unsigned compositeTexture, std::vector<uint8_t>& rgbaOut)
{
    if (gl_ == nullptr || fbo_ == 0)
        return false;

    constexpr int kScopeBytes = kScopeW * kScopeH * 4;
    if (scopeTex_ == 0)
    {
        glGenTextures (1, &scopeTex_);
        glBindTexture (GL_TEXTURE_2D, scopeTex_);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, kScopeW, kScopeH, 0,
                      GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        gl_->GenBuffers (2, scopePbo_);
        for (int i = 0; i < 2; ++i)
        {
            gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, scopePbo_[i]);
            gl_->BufferData (GL_PIXEL_PACK_BUFFER, kScopeBytes, nullptr, GL_STREAM_READ);
        }
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
        scopePending_[0] = scopePending_[1] = false;
        scopeIdx_ = 0;
    }

    bool have = false;
    const int prev = scopeIdx_ ^ 1;
    if (scopePending_[prev])
    {
        gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, scopePbo_[prev]);
        if (const void* p = gl_->MapBufferRange (GL_PIXEL_PACK_BUFFER, 0,
                                                 kScopeBytes, GL_MAP_READ_BIT))
        {
            rgbaOut.resize ((size_t) kScopeBytes);
            std::memcpy (rgbaOut.data(), p, (size_t) kScopeBytes);
            have = true;
            gl_->UnmapBuffer (GL_PIXEL_PACK_BUFFER);
        }
        scopePending_[prev] = false;
    }

    // Downscale the composite into the scope target (same blit program as
    // presentToWindow, into the FBO) and issue the next async readback.
    gl_->BindFramebuffer (GL_FRAMEBUFFER, fbo_);
    attachTarget (scopeTex_);
    glViewport (0, 0, kScopeW, kScopeH);
    glDisable (GL_BLEND);
    gl_->UseProgram (blit_.prog);
    bindTexture (0, compositeTexture, blit_.uTexture);
    drawQuad();

    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, scopePbo_[scopeIdx_]);
    glPixelStorei (GL_PACK_ALIGNMENT, 1);
    glReadPixels (0, 0, kScopeW, kScopeH, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
    scopePending_[scopeIdx_] = true;
    gl_->BindBuffer (GL_PIXEL_PACK_BUFFER, 0);
    gl_->BindFramebuffer (GL_FRAMEBUFFER, 0);
    scopeIdx_ = prev;
    return have;
}

} // namespace videorender

#endif // ARBIT_HAVE_VIEWPORT
