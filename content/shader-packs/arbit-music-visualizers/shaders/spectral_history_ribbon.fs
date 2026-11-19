/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "A mirrored spectrum ribbon that writes the current mix into a persistent scrolling history. The center line carries low frequencies and the outer peaks carry the upper bands.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Audio Visualizer", "Feedback", "Generator"],
  "INPUTS": [
    { "NAME": "gain", "TYPE": "float", "DEFAULT": 1.35, "MIN": 0.1, "MAX": 4.0, "LABEL": "Spectrum gain" },
    { "NAME": "scrollSpeed", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.1, "MAX": 4.0, "LABEL": "History speed" },
    { "NAME": "decay", "TYPE": "float", "DEFAULT": 0.985, "MIN": 0.90, "MAX": 1.0, "LABEL": "History decay" },
    { "NAME": "lineWidth", "TYPE": "float", "DEFAULT": 0.012, "MIN": 0.002, "MAX": 0.05, "LABEL": "Ribbon width" },
    { "NAME": "lowColor", "TYPE": "color", "DEFAULT": [0.08, 0.82, 1.0, 1.0], "LABEL": "Low-band color" },
    { "NAME": "highColor", "TYPE": "color", "DEFAULT": [1.0, 0.38, 0.16, 1.0], "LABEL": "High-band color" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.002, 0.002, 0.006, 1.0], "LABEL": "Background" }
  ],
  "PASSES": [
    { "TARGET": "history", "PERSISTENT": true },
    {}
  ]
}*/

float pixelMask(float distanceValue, float halfWidth)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(halfWidth - aa, halfWidth + aa, distanceValue);
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    if (PASSINDEX == 0)
    {
        float beatsThisFrame = max(TIMEDELTA, 0.0) * max(uBPM, 1.0) / 60.0;
        float shift = scrollSpeed * beatsThisFrame * float(uPlaying) * 0.22;
        vec2 historyUv = vec2(uv.x + shift, uv.y);
        vec3 previous = historyUv.x <= 1.0
            ? texture(history, historyUv).rgb * decay
            : backgroundColor.rgb;

        float bandPosition = clamp(abs(uv.y - 0.5) * 2.0, 0.0, 1.0);
        float band = pow(bandPosition, 1.55) * 63.0;
        float amplitude = clamp(arbitBand(band) * gain, 0.0, 1.0);
        float target = amplitude * 0.46;
        float ribbonDistance = abs(abs(uv.y - 0.5) - target);
        float halfWidth = max(lineWidth * 0.5, 1.15 / minDimension);
        float core = pixelMask(ribbonDistance, halfWidth);
        float rim = max(0.0, pixelMask(ribbonDistance,
                        halfWidth + 1.25 / minDimension) - core);
        float halo = exp(-max(ribbonDistance - halfWidth, 0.0)
                         * minDimension / 6.0) * (1.0 - core);
        float injection = smoothstep(0.965, 0.995, uv.x);
        vec3 ribbonColor = mix(lowColor.rgb, highColor.rgb, bandPosition);
        float energy = 0.45 + 0.8 * uRMS + 0.5 * uOnset;
        vec3 current = (vec3(1.12) * core
                       + ribbonColor * (core * 0.65 + rim * 1.45 + halo * 0.22))
                       * energy * injection;

        float centerDistance = abs(uv.y - 0.5);
        float center = pixelMask(centerDistance, 0.65 / minDimension)
                     + exp(-centerDistance * minDimension / 4.0) * 0.18;
        current += lowColor.rgb * center * injection * (0.15 + 0.5 * uRMS);
        gl_FragColor = vec4(max(previous, current), 1.0);
    }
    else
    {
        vec3 value = texture(history, uv).rgb;
        value += backgroundColor.rgb;
        value = value / (value + vec3(0.8));
        gl_FragColor = vec4(pow(max(value, 0.0), vec3(0.82)), 1.0);
    }
}
