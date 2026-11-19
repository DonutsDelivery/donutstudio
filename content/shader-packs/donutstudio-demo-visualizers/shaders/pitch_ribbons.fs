/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "A horizontal score window rendered as luminous pitch ribbons. Past notes trail left, upcoming notes approach from the right, and exact harmonic masters connect related voices.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Score", "Voice Leading", "Generator"],
  "INPUTS": [
    { "NAME": "historyBeats", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0, "MAX": 16.0, "LABEL": "History beats" },
    { "NAME": "lookaheadBeats", "TYPE": "float", "DEFAULT": 8.0, "MIN": 1.0, "MAX": 16.0, "LABEL": "Lookahead beats" },
    { "NAME": "ribbonWidth", "TYPE": "float", "DEFAULT": 0.014, "MIN": 0.003, "MAX": 0.05, "LABEL": "Ribbon width" },
    { "NAME": "linkStrength", "TYPE": "float", "DEFAULT": 0.55, "MIN": 0.0, "MAX": 1.5, "LABEL": "Harmonic links" },
    { "NAME": "lowColor", "TYPE": "color", "DEFAULT": [0.06, 0.72, 1.0, 1.0], "LABEL": "Low-note color" },
    { "NAME": "highColor", "TYPE": "color", "DEFAULT": [1.0, 0.25, 0.68, 1.0], "LABEL": "High-note color" },
    { "NAME": "rootColor", "TYPE": "color", "DEFAULT": [1.0, 0.74, 0.18, 1.0], "LABEL": "Root color" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.004, 0.006, 0.014, 1.0], "LABEL": "Background" }
  ]
}*/

float segmentDistance(vec2 point, vec2 start, vec2 end)
{
    vec2 line = end - start;
    float amount = clamp(dot(point - start, line) / max(dot(line, line), 1e-6), 0.0, 1.0);
    return length(point - start - line * amount);
}

float pixelMask(float distanceValue, float halfWidth)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(halfWidth - aa, halfWidth + aa, distanceValue);
}

float exactMidi(ArbitNote note)
{
    float frequency = max(note.t1.x, 1.0);
    return 69.0 + 12.0 * log2(frequency / 440.0);
}

float midiY(float midi)
{
    return clamp((midi - 21.0) / 87.0, 0.02, 0.98);
}

float pitchY(ArbitNote note)
{
    return midiY(exactMidi(note));
}

float beatX(float offset)
{
    float span = max(historyBeats + lookaheadBeats, 0.01);
    return clamp((offset + historyBeats) / span, -0.1, 1.1);
}

float noteVisibility(ArbitNote note)
{
    float start = -note.t0.z;
    float end = note.t0.w;
    if (start > lookaheadBeats || end < -historyBeats) return 0.0;
    float edgeFade = smoothstep(-historyBeats, -historyBeats + 0.5, end)
                   * (1.0 - smoothstep(lookaheadBeats - 0.5, lookaheadBeats, start));
    return edgeFade;
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec3 color = backgroundColor.rgb;
    float playhead = beatX(0.0);
    float playheadDistance = abs(uv.x - playhead);
    color += vec3(0.12, 0.26, 0.48)
           * (pixelMask(playheadDistance, 0.8 / minDimension)
              + exp(-playheadDistance * minDimension / 4.0) * 0.22)
           * (0.5 + beatPulse(0.15));

    for (int edge = 0; edge < uLinkCount; ++edge)
    {
        ArbitLink link = arbitLink(edge);
        ArbitNote slave = arbitNote(int(link.t0.x + 0.5));
        ArbitNote master = arbitNote(int(link.t0.y + 0.5));
        float visibility = min(noteVisibility(slave), noteVisibility(master));
        if (visibility <= 0.001) continue;
        vec2 start = vec2(beatX(-slave.t0.z), pitchY(slave));
        vec2 end = vec2(beatX(-master.t0.z), pitchY(master));
        float distanceToLink = segmentDistance(uv, start, end);
        float linkHalfWidth = max(ribbonWidth * 0.16, 0.8 / minDimension);
        float linkCore = pixelMask(distanceToLink, linkHalfWidth);
        float linkHalo = exp(-max(distanceToLink - linkHalfWidth, 0.0)
                             * minDimension / 4.5) * (1.0 - linkCore);
        color += rootColor.rgb
               * (linkCore * 0.85 + linkHalo * 0.18)
               * visibility * linkStrength * 0.35;
    }

    for (int row = 0; row < uNoteCount; ++row)
    {
        ArbitNote note = arbitNote(row);
        if (note.t0.y <= 0.0) continue;
        float visibility = noteVisibility(note);
        if (visibility <= 0.001) continue;
        vec2 start = vec2(beatX(-note.t0.z), pitchY(note));
        vec2 end = vec2(beatX(note.t0.w), pitchY(note));
        float distanceToRibbon = segmentDistance(uv, start, end);
        float pitch = pitchY(note);
        float soundingMidi = exactMidi(note);
        float equalY = midiY(floor(soundingMidi + 0.5));
        float detune = clamp(abs(soundingMidi - floor(soundingMidi + 0.5)) * 2.0, 0.0, 1.0);
        vec2 equalStart = vec2(start.x, equalY);
        vec2 equalEnd = vec2(end.x, equalY);
        float equalGuide = pixelMask(segmentDistance(uv, equalStart, equalEnd),
                                     0.65 / minDimension);
        float tuningTether = pixelMask(segmentDistance(uv, start, equalStart),
                                       0.55 / minDimension);
        color += mix(vec3(0.20, 0.62, 0.92), rootColor.rgb, clamp(note.t1.w, 0.0, 1.0))
               * (equalGuide * 0.28 + tuningTether * 0.42) * detune * visibility;
        vec3 ribbonColor = mix(lowColor.rgb, highColor.rgb, pitch);
        ribbonColor = mix(ribbonColor, rootColor.rgb, clamp(note.t1.w, 0.0, 1.0));
        ribbonColor = mix(ribbonColor, vec3(0.22, 1.0, 0.82), detune * 0.30);
        float halfWidth = max(ribbonWidth * 0.5, 1.2 / minDimension);
        float core = pixelMask(distanceToRibbon, halfWidth);
        float rim = max(0.0, pixelMask(distanceToRibbon,
                        halfWidth + 1.35 / minDimension) - core);
        float halo = exp(-max(distanceToRibbon - halfWidth, 0.0)
                         * minDimension / 6.0) * (1.0 - core);
        float sounding = arbitNoteActive(row);
        color += (vec3(1.15) * core
                  + ribbonColor * (core * 0.72 + rim * 1.6 + halo * 0.20)) * visibility
               * (0.5 + note.t0.y + sounding * (0.5 + uOnset));
    }

    color = color / (color + vec3(0.78));
    gl_FragColor = vec4(pow(max(color, 0.0), vec3(0.84)), 1.0);
}
