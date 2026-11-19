/*{
  "DESCRIPTION": "Apollonian Harmonic Gasket — an iterated circle-inversion fractal whose nested arcs lock into a self-similar gasket when the per-fold scale is a pure just-intonation ratio (num/den) and precess/smear when the ratio is detuned in cents. The figure is a calm standing structure that breathes with the beat and brightens with onsets; route HarmRatioNum/HarmRatioDen onto ratioP/ratioQ, CentsFromRoot onto detune, and audio/clock onto glow/zoom/pulse. The Arbit thesis made visible: pure ratios tile perfectly, temperament fractures the lines. Original work for the Arbit Media Machine. License: CC0.",
  "CREDIT": "Arbit",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Geometry"],
  "INPUTS": [
    { "NAME": "ratioP",    "TYPE": "float", "DEFAULT": 3.0,  "MIN": 1.0,  "MAX": 16.0 },
    { "NAME": "ratioQ",    "TYPE": "float", "DEFAULT": 2.0,  "MIN": 1.0,  "MAX": 16.0 },
    { "NAME": "symmetry",  "TYPE": "float", "DEFAULT": 5.0,  "MIN": 3.0,  "MAX": 9.0 },
    { "NAME": "detune",    "TYPE": "float", "DEFAULT": 0.0,  "MIN": -50.0, "MAX": 50.0 },
    { "NAME": "octaves",   "TYPE": "float", "DEFAULT": 5.0,  "MIN": 2.0,  "MAX": 9.0 },
    { "NAME": "zoom",      "TYPE": "float", "DEFAULT": 1.0,  "MIN": 0.35, "MAX": 3.0 },
    { "NAME": "spin",      "TYPE": "float", "DEFAULT": 0.05, "MIN": -1.0, "MAX": 1.0 },
    { "NAME": "lineWidth", "TYPE": "float", "DEFAULT": 0.02, "MIN": 0.004, "MAX": 0.08 },
    { "NAME": "pulse",     "TYPE": "float", "DEFAULT": 0.0,  "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "glow",      "TYPE": "float", "DEFAULT": 0.6,  "MIN": 0.0,  "MAX": 2.0 },
    { "NAME": "hue",       "TYPE": "float", "DEFAULT": 0.58, "MIN": 0.0,  "MAX": 1.0 },
    { "NAME": "brightness","TYPE": "float", "DEFAULT": 1.0,  "MIN": 0.2,  "MAX": 2.0 }
  ]
}*/

#define PI  3.14159265359
#define TAU 6.28318530718

// Rotate a 2D vector by angle a.
vec2 rot2(vec2 v, float a) {
    float s = sin(a), c = cos(a);
    return vec2(c * v.x - s * v.y, s * v.x + c * v.y);
}

void main() {
    // Aspect-correct, centred coordinates (RENDERSIZE == uResolution).
    vec2 uv = (gl_FragCoord.xy - 0.5 * RENDERSIZE.xy) / RENDERSIZE.y;

    // ---- Harmonic ratio drives the whole figure ----------------------------
    // ratioP/ratioQ are HarmRatioNum/HarmRatioDen. A pure JI ratio gives a
    // per-fold scale that nests EXACTLY (the gasket tiles); detune (cents off
    // the root) multiplies it by 2^(cents/1200) so temperament drifts the scale
    // off the lattice and the nested arcs precess instead of locking.
    float p = max(ratioP, 1.0);
    float q = max(ratioQ, 1.0);
    float ratio = p / q;                              // e.g. 3/2 = perfect fifth
    float cents = clamp(detune, -50.0, 50.0);
    float bend  = exp2(cents / 1200.0);               // pure ratio when cents == 0
    // Fold scale lives in a calm band so the figure stays a standing structure.
    float foldScale = mix(0.62, 0.92, fract(log2(max(ratio, 1.0001)))) * bend;
    foldScale = clamp(foldScale, 0.55, 0.97);

    // Detune also offsets the inversion centre off-axis: zero when locked, so
    // pure ratios sit dead-still and only cents-off "smears" the centres.
    float off = (bend - 1.0) * 6.0;                   // ~0 at pure JI
    vec2 invCentre = vec2(off, 0.0);

    // ---- View transform: breathe on the beat, zoom/spin from mods ----------
    float breathe = 1.0 + 0.04 * sin(uBeatPhase * TAU) + 0.06 * pulse * beatPulse(0.35);
    vec2 z = uv * (1.0 / max(zoom, 1e-3)) * breathe;
    z = rot2(z, uTime * spin * 0.5);

    // Number of inversion centres = symmetry wedges (rounded to a stable int).
    float sym = clamp(floor(symmetry + 0.5), 3.0, 9.0);
    float wedge = TAU / sym;

    // ---- Iterated fold + circle inversion (Apollonian gasket skeleton) ------
    // Each iteration: kaleidoscopic fold into one wedge, invert about a unit
    // circle offset by invCentre, then rescale by foldScale. The distance from
    // the inversion rim, tracked as the running minimum, is the gasket's
    // self-similar arc network — the LINES the user wants.
    int N = int(clamp(floor(octaves + 0.5), 2.0, 9.0));
    float dLine = 1e9;     // nearest arc (rim of an inversion circle)
    float dNode = 1e9;     // nearest tangency node (gasket vertices)
    float depthHit = 0.0;  // iteration depth where the nearest arc was found
    float scl = 1.0;       // accumulated scale (keeps line width metric-correct)

    for (int i = 0; i < 9; i++) {
        if (i >= N) break;

        // Kaleidoscopic fold: mirror into a single wedge so the gasket is
        // 'sym'-fold symmetric (ordered, not chaotic).
        float ang = atan(z.y, z.x);
        ang = mod(ang, wedge);
        ang = abs(ang - 0.5 * wedge);                 // mirror within the wedge
        float rad = length(z);
        z = vec2(cos(ang), sin(ang)) * rad;

        // Circle inversion about (1) the unit circle, offset by the detune.
        vec2 c = z - invCentre;
        float r2 = max(dot(c, c), 1e-4);
        // Distance to this inversion circle's RIM (|c| == 1) → the arc/line.
        float rim = abs(length(c) - 1.0) * scl;
        if (rim < dLine) { dLine = rim; depthHit = float(i); }
        // Tangency node where the folded ray meets the wedge spine.
        dNode = min(dNode, length(z - vec2(cos(0.5 * wedge), sin(0.5 * wedge))) * scl);

        // Invert (the conformal map that makes the gasket self-similar) then
        // rescale toward the centre so the next octave nests inside.
        z = invCentre + c / r2;
        z *= foldScale;
        scl *= foldScale;
    }

    // ---- Shade the arcs as glowing lines -----------------------------------
    float lw = clamp(lineWidth, 0.004, 0.08);
    // Vibration: the arcs ripple along time + harmonic ratio. Pure ratios keep
    // a steady standing wave; detune adds a slow drift term (the smear).
    float vib = 0.5 + 0.5 * sin(uTime * (2.0 + ratio) + depthHit * 1.7);
    float smear = abs(cents) / 50.0;                  // 0 locked … 1 fully off
    float lineW = lw * (1.0 + 0.6 * vib) * (1.0 + 1.2 * smear);

    float line = 1.0 - smoothstep(0.0, lineW, dLine);
    float halo = (1.0 - smoothstep(0.0, lineW * 8.0, dLine)) * 0.35;
    float node = 1.0 - smoothstep(0.0, lineW * 1.6, dNode);

    // Depth-graded hue: deeper octaves shift hue slightly so nested scales read
    // as distinct rings of the same family (musical, ordered).
    float depthN = depthHit / max(float(N) - 1.0, 1.0);
    float h = fract(hue + 0.06 * depthN + 0.04 * vib);
    float sat = 0.55 + 0.35 * (1.0 - smear);          // pure = saturated, detune = washed
    vec3 lineCol = hsv2rgb(vec3(h, sat, 1.0));
    vec3 nodeCol = hsv2rgb(vec3(fract(h + 0.5), 0.4, 1.0));

    // Background: a calm dark gradient that lifts faintly with loudness.
    vec3 col = mix(vec3(0.010, 0.013, 0.022), vec3(0.018, 0.016, 0.030),
                   0.5 + 0.5 * uv.y);
    col += vec3(0.02, 0.025, 0.04) * uRMS;

    // Energy: clock breathing + audio onset + the pulse mod, scaled by glow.
    float energy = 0.55 + 0.55 * uRMS + 0.9 * uOnset + 0.5 * pulse;
    col += lineCol * (line + halo) * energy * (0.5 + glow);
    col += nodeCol * node * (0.6 + 0.8 * uOnset);

    // Onset flash ripples outward from centre as a brief ring (musical accent).
    float rr = length(uv);
    float ring = exp(-pow((rr - uOnset * 0.9) * 7.0, 2.0)) * uOnset;
    col += hsv2rgb(vec3(fract(hue + 0.5), 0.3, 1.0)) * ring * 0.6;

    col *= brightness;
    // Gentle filmic-ish lift; keep it calm (no harsh blowout).
    col = pow(col, vec3(0.92));

    gl_FragColor = arbitGuard(vec4(col, 1.0));
}
