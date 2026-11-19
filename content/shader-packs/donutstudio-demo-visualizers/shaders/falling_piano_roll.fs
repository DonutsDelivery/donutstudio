/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "A falling piano roll driven by DonutStudio score lookahead. Exact note frequency shifts microtonal notes between equal-tempered lanes. Notes strike a keyboard with crisp luminous cores, deterministic surface texture, glow, and impact sparks.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Score", "Music Visualizer", "Generator"],
  "INPUTS": [
    { "NAME": "lookaheadBeats", "TYPE": "float", "DEFAULT": 8.0, "MIN": 2.0, "MAX": 16.0, "LABEL": "Lookahead beats" },
    { "NAME": "keyboardHeight", "TYPE": "float", "DEFAULT": 0.18, "MIN": 0.08, "MAX": 0.32, "LABEL": "Keyboard height" },
    { "NAME": "noteWidth", "TYPE": "float", "DEFAULT": 0.72, "MIN": 0.25, "MAX": 1.0, "LABEL": "Note width" },
    { "NAME": "glowAmount", "TYPE": "float", "DEFAULT": 0.85, "MIN": 0.0, "MAX": 2.0, "LABEL": "Glow" },
    { "NAME": "sparkAmount", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.0, "MAX": 2.0, "LABEL": "Impact sparks" },
    { "NAME": "lowColor", "TYPE": "color", "DEFAULT": [0.08, 0.55, 1.0, 1.0], "LABEL": "Low-note color" },
    { "NAME": "highColor", "TYPE": "color", "DEFAULT": [0.92, 0.18, 1.0, 1.0], "LABEL": "High-note color" },
    { "NAME": "rootColor", "TYPE": "color", "DEFAULT": [1.0, 0.72, 0.16, 1.0], "LABEL": "Root color" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.006, 0.008, 0.018, 1.0], "LABEL": "Background" },
    { "NAME": "showKeyboard", "TYPE": "bool", "DEFAULT": true, "LABEL": "Show keyboard" }
  ]
}*/

float hash11(float p)
{
    return fract(sin(p * 127.1 + 311.7) * 43758.5453);
}

float hash21(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

float roundedBoxDistance(vec2 p, vec2 halfSize, float radius)
{
    vec2 q = abs(p) - halfSize + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float pixelMask(float distanceValue)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(-aa, aa, distanceValue);
}

float exactMidi(ArbitNote note)
{
    float midi = note.t0.x;
    float equalHz = 440.0 * pow(2.0, (midi - 69.0) / 12.0);
    return midi + 12.0 * log2(max(note.t1.x, 0.001) / max(equalHz, 0.001));
}

vec3 noteColor(ArbitNote note, float lane)
{
    vec3 color = mix(lowColor.rgb, highColor.rgb, lane);
    float trackShift = fract(note.t1.z * 0.173);
    color = mix(color, hsv2rgb(vec3(trackShift, 0.74, 1.0)), 0.24);
    return mix(color, rootColor.rgb, clamp(note.t1.w, 0.0, 1.0));
}

float pianoStepCoordinate(float midiStep)
{
    float octaveBase = floor(midiStep / 12.0) * 7.0 - 12.0;
    float pc = mod(midiStep, 12.0);
    float offset = 0.5;
    if (pc == 1.0) offset = 1.0;
    else if (pc == 2.0) offset = 1.5;
    else if (pc == 3.0) offset = 2.0;
    else if (pc == 4.0) offset = 2.5;
    else if (pc == 5.0) offset = 3.5;
    else if (pc == 6.0) offset = 4.0;
    else if (pc == 7.0) offset = 4.5;
    else if (pc == 8.0) offset = 5.0;
    else if (pc == 9.0) offset = 5.5;
    else if (pc == 10.0) offset = 6.0;
    else if (pc == 11.0) offset = 6.5;
    return octaveBase + offset;
}

float pianoCoordinate(float midi)
{
    float lo = floor(midi);
    return mix(pianoStepCoordinate(lo), pianoStepCoordinate(lo + 1.0), fract(midi));
}

float hasBlackKeyAfter(float whiteIndex)
{
    float degree = mod(whiteIndex, 7.0);
    return (degree == 0.0 || degree == 2.0 || degree == 3.0
         || degree == 5.0 || degree == 6.0) ? 1.0 : 0.0;
}

vec3 filmic(vec3 value)
{
    value = max(value, 0.0);
    return clamp((value * (2.51 * value + 0.03))
               / (value * (2.43 * value + 0.59) + 0.14), 0.0, 1.0);
}

void main()
{
    // The shader contract presents display-space coordinates with y=0 at the
    // bottom, independent of the compositor's top-down framebuffer storage.
    vec2 uv = isf_FragNormCoord;
    float lowMidi = 21.0;
    float highMidi = 108.0;
    float whiteKeys = 52.0;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec3 color = backgroundColor.rgb;

    float horizonDistance = abs(uv.y - keyboardHeight);
    float horizonFog = exp(-horizonDistance * 19.0);
    float centreVignette = 1.0 - smoothstep(0.2, 0.82, length((uv - vec2(0.5, 0.48)) * vec2(0.86, 1.0)));
    float verticalDepth = pow(max(0.0, 1.0 - uv.y), 1.4);
    color += vec3(0.012, 0.035, 0.070) * centreVignette;
    color += vec3(0.018, 0.030, 0.070) * verticalDepth * (0.45 + 0.55 * beatPulse(0.12));
    color += mix(rootColor.rgb, vec3(0.06, 0.25, 0.42), 0.55)
           * horizonFog * (0.035 + 0.035 * glowAmount);

    // Fine lane depth gives the falling notes a place in space without turning
    // the background into a visible grid.
    float laneLine = 1.0 - smoothstep(0.0, 1.25,
        abs(fract(uv.x * whiteKeys) - 0.5) * minDimension / whiteKeys);
    color += vec3(0.02, 0.04, 0.07) * laneLine * smoothstep(keyboardHeight, 1.0, uv.y) * 0.22;

    if (showKeyboard && uv.y < keyboardHeight)
    {
        float whiteCoordinate = uv.x * whiteKeys;
        float localX = fract(whiteCoordinate) - 0.5;
        vec3 whiteKey = vec3(0.84, 0.87, 0.92);
        vec3 blackKey = vec3(0.018, 0.023, 0.036);
        float edgeDistance = (0.48 - abs(localX)) * RENDERSIZE.x / whiteKeys;
        float keyBody = smoothstep(0.0, 1.0, edgeDistance);
        float verticalShade = mix(0.72, 1.06, clamp(uv.y / max(keyboardHeight, 0.001), 0.0, 1.0));
        color = mix(vec3(0.012), whiteKey * verticalShade, keyBody);
        color += vec3(0.04, 0.045, 0.06)
               * (1.0 - smoothstep(0.0, 1.4, abs(localX) * RENDERSIZE.x / whiteKeys));

        float boundary = floor(whiteCoordinate + 0.5);
        float leftWhiteIndex = boundary - 1.0;
        float blackDistance = abs(whiteCoordinate - boundary);
        float blackMask = hasBlackKeyAfter(leftWhiteIndex)
            * (1.0 - smoothstep(0.27, 0.31, blackDistance))
            * smoothstep(keyboardHeight * 0.34, keyboardHeight * 0.37, uv.y);
        color = mix(color, blackKey * mix(0.72, 1.08,
            clamp(uv.y / max(keyboardHeight, 0.001), 0.0, 1.0)), blackMask);

        float horizonPixels = horizonDistance * minDimension;
        color += rootColor.rgb * exp(-horizonPixels / 10.0) * (0.75 + glowAmount * 0.28);
        color += vec3(1.35, 1.05, 0.64) * exp(-horizonPixels / 1.15);
    }

    float visibleHeight = max(0.2, 1.0 - keyboardHeight);
    for (int i = 0; i < uNoteCount; ++i)
    {
        ArbitNote note = arbitNote(i);
        if (note.t0.y <= 0.0) continue;

        float midi = exactMidi(note);
        float lane = pianoCoordinate(midi) / whiteKeys;
        if (lane < -0.02 || lane > 1.02) continue;

        float startOffset = arbitNoteStartOffset(i);
        float endOffset = arbitNoteEndOffset(i);
        if (startOffset > lookaheadBeats || endOffset < -0.8) continue;

        float y0 = keyboardHeight + startOffset / lookaheadBeats * visibleHeight;
        float y1 = keyboardHeight + endOffset / lookaheadBeats * visibleHeight;
        float clippedY0 = max(keyboardHeight, min(y0, 1.1));
        float clippedY1 = max(keyboardHeight, min(y1, 1.1));
        if (clippedY1 <= clippedY0) continue;

        float laneWidth = mix(0.28, 0.82, noteWidth) / whiteKeys;
        vec2 center = vec2(lane, 0.5 * (clippedY0 + clippedY1));
        vec2 halfSize = vec2(laneWidth * 0.5,
                             max(1.25 / minDimension, 0.5 * (clippedY1 - clippedY0)));
        float corner = min(5.0 / minDimension, laneWidth * 0.34);
        float distanceValue = roundedBoxDistance(uv - center, halfSize, corner);
        float body = pixelMask(distanceValue);
        float insetDistance = distanceValue + 1.45 / minDimension;
        float core = pixelMask(insetDistance);
        float rim = max(0.0, body - core);
        float outsidePixels = max(distanceValue, 0.0) * minDimension;
        float glowRadius = mix(1.5, 7.0, clamp(glowAmount * 0.5, 0.0, 1.0));
        float glow = exp(-outsidePixels / glowRadius) * (1.0 - body);

        vec3 noteRgb = noteColor(note,
            clamp((midi - lowMidi) / (highMidi - lowMidi), 0.0, 1.0));
        vec2 textureCell = floor((uv - center + halfSize) * RENDERSIZE.xy * 0.31);
        float fleckNoise = hash21(textureCell + vec2(float(i) * 13.7, float(i) * 5.3));
        float fleck = core * step(0.74, fleckNoise) * smoothstep(4.0, 12.0, halfSize.y * minDimension);
        float velocityLight = 0.72 + clamp(note.t0.y, 0.0, 1.0) * 0.72;

        color += noteRgb * glow * (0.58 + glowAmount * 0.52);
        color += noteRgb * rim * 2.2;
        color += core * (vec3(1.35) + noteRgb * 1.25) * velocityLight;
        color -= fleck * mix(noteRgb * 0.55, vec3(0.72), 0.45);

        // A narrow light shaft behind each body adds depth while retaining a
        // hard note silhouette.
        float shaftX = exp(-abs(uv.x - lane) * RENDERSIZE.x / 10.0);
        float shaftY = smoothstep(keyboardHeight, clippedY1, uv.y)
                     * (1.0 - smoothstep(clippedY1, min(1.0, clippedY1 + 0.16), uv.y));
        color += noteRgb * shaftX * shaftY * 0.045 * glowAmount;

        float age = note.t0.z;
        if (age >= 0.0 && age < 0.8)
        {
            float impact = 1.0 - age / 0.8;
            float hit = exp(-abs(uv.x - lane) * RENDERSIZE.x / 13.0)
                      * exp(-abs(uv.y - keyboardHeight) * minDimension / 7.0);
            color += noteRgb * hit * impact * (1.3 + glowAmount * 0.8);
            color += vec3(1.4) * hit * hit * impact;

            for (int spark = 0; spark < 9; ++spark)
            {
                float seed = float(i * 17 + spark * 41);
                float direction = mix(-1.0, 1.0, hash11(seed));
                float rise = mix(0.028, 0.14, hash11(seed + 9.0));
                float spread = mix(0.008, 0.055, hash11(seed + 23.0));
                vec2 pos = vec2(lane + direction * spread * age,
                                keyboardHeight + rise * age - 0.055 * age * age);
                float radiusPixels = mix(0.65, 1.8, hash11(seed + 37.0));
                float sparkDistance = distance(uv, pos) * minDimension;
                float sparkCore = 1.0 - smoothstep(radiusPixels, radiusPixels + 1.0, sparkDistance);
                float sparkGlow = exp(-sparkDistance / (radiusPixels * 2.8));
                color += (vec3(1.2) * sparkCore + noteRgb * sparkGlow * 0.8)
                       * impact * sparkAmount;
            }
        }
    }

    float vignette = 1.0 - 0.34 * smoothstep(0.45, 0.92,
        length((uv - 0.5) * vec2(0.82, 1.0)));
    color = filmic(color * vignette * 1.08);
    gl_FragColor = vec4(pow(max(color, 0.0), vec3(0.94)), 1.0);
}
