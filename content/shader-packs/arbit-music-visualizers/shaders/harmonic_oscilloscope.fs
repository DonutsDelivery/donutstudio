/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "A root-locked score oscilloscope. One phosphor trace combines every sounding note at its exact frequency ratio to the project root; optional free-run drift releases the trigger lock.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Score", "Just Intonation", "Oscilloscope", "Generator"],
  "INPUTS": [
    { "NAME": "windowCycles", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0, "MAX": 12.0, "LABEL": "Root cycles" },
    { "NAME": "amplitude", "TYPE": "float", "DEFAULT": 0.78, "MIN": 0.10, "MAX": 1.0, "LABEL": "Wave amplitude" },
    { "NAME": "traceWidth", "TYPE": "float", "DEFAULT": 0.0024, "MIN": 0.001, "MAX": 0.025, "LABEL": "Trace width" },
    { "NAME": "glowAmount", "TYPE": "float", "DEFAULT": 0.72, "MIN": 0.0, "MAX": 2.0, "LABEL": "Phosphor glow" },
    { "NAME": "waveShape", "TYPE": "float", "DEFAULT": 0.66, "MIN": 0.0, "MAX": 1.0, "LABEL": "Harmonic shape" },
    { "NAME": "phaseDrift", "TYPE": "float", "DEFAULT": 0.0, "MIN": 0.0, "MAX": 1.5, "LABEL": "Free-run drift" },
    { "NAME": "sampleCount", "TYPE": "float", "DEFAULT": 320.0, "MIN": 64.0, "MAX": 1024.0, "LABEL": "Scope samples" },
    { "NAME": "showComponents", "TYPE": "bool", "DEFAULT": false, "LABEL": "Stack note channels" },
    { "NAME": "traceColor", "TYPE": "color", "DEFAULT": [0.72, 0.94, 1.0, 1.0], "LABEL": "Phosphor trace" },
    { "NAME": "rootColor", "TYPE": "color", "DEFAULT": [1.0, 0.68, 0.16, 1.0], "LABEL": "Root channel" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.005, 0.011, 0.014, 1.0], "LABEL": "Scope face" }
  ]
}*/

#define TAU 6.28318530718

float pixelMask(float distanceValue, float halfWidth)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(halfWidth - aa, halfWidth + aa, distanceValue);
}

float noteRatio(ArbitNote note)
{
    return max(note.t1.x, 1.0) / max(uRootFreq, 1.0);
}

float oscillator(float cycles)
{
    float phase = TAU * cycles;
    float sine = sin(phase);
    float cosine = cos(phase);
    float sine2 = 2.0 * sine * cosine;
    float sineSquared = sine * sine;
    float sine3 = sine * (3.0 - 4.0 * sineSquared);
    float sine5 = sine * (16.0 * sineSquared * sineSquared - 20.0 * sineSquared + 5.0);
    float rich = sine
               + sine2 * 0.34
               + sine3 * 0.18
               + sine5 * 0.08;
    return mix(sine, rich / 1.48, waveShape);
}

float scopeSample(float sweepPosition)
{
    float combined = 0.0;
    float totalWeight = 0.0;
    for (int i = 0; i < uNoteCount; ++i)
    {
        ArbitNote note = arbitNote(i);
        float visibility = arbitNoteActive(i);
        if (visibility <= 0.001) continue;
        float velocity = clamp(note.t0.y, 0.0, 1.0);
        float weight = visibility * (0.30 + velocity * 0.70);
        float ratio = noteRatio(note);
        float relativePhase = uBeat * phaseDrift * ratio;
        combined += oscillator(sweepPosition * windowCycles * ratio + relativePhase) * weight;
        totalWeight += weight;
    }

    if (totalWeight > 0.001)
    {
        float signal = combined / totalWeight;
        return clamp(signal / (0.70 + abs(signal) * 0.30), -1.0, 1.0);
    }

    float bass = clamp(arbitBand(4.0) * 4.0, 0.0, 1.0);
    float mids = clamp(arbitBand(20.0) * 4.0, 0.0, 1.0);
    float treble = clamp(arbitBand(50.0) * 5.0, 0.0, 1.0);
    return oscillator(sweepPosition * windowCycles + uBeat * 0.04) * (0.24 + bass * 0.58)
         + oscillator(sweepPosition * windowCycles * 2.0 - uBeat * 0.07) * mids * 0.22
         + sin(TAU * sweepPosition * windowCycles * 7.0) * treble * 0.08;
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec3 color = backgroundColor.rgb;

    for (int grid = 1; grid < 10; ++grid)
    {
        float distanceToGrid = abs(uv.x - float(grid) / 10.0);
        color += vec3(0.035, 0.075, 0.070)
               * pixelMask(distanceToGrid, 0.48 / minDimension);
    }
    for (int grid = 1; grid < 8; ++grid)
    {
        float distanceToGrid = abs(uv.y - float(grid) / 8.0);
        color += vec3(0.035, 0.075, 0.070)
               * pixelMask(distanceToGrid, 0.48 / minDimension);
    }
    float axisDistance = abs(uv.y - 0.5);
    color += vec3(0.25, 0.055, 0.075)
           * pixelMask(axisDistance, 0.62 / minDimension);

    if (showComponents)
    {
        for (int i = 0; i < uNoteCount; ++i)
        {
            ArbitNote note = arbitNote(i);
            float visibility = arbitNoteActive(i);
            if (visibility <= 0.001) continue;
            float ratio = noteRatio(note);
            float channel = fract(log2(ratio) + 0.035 * float(i));
            float baseline = 0.12 + channel * 0.76;
            float value = oscillator(uv.x * windowCycles * ratio
                                     + uBeat * phaseDrift * ratio);
            float channelY = baseline + value * 0.025;
            float distanceToChannel = abs(uv.y - channelY);
            vec3 channelColor = hsv2rgb(vec3(fract(log2(ratio) * 0.38 + 0.55), 0.62, 1.0));
            channelColor = mix(channelColor, rootColor.rgb, clamp(note.t1.w, 0.0, 1.0));
            color += channelColor
                   * (pixelMask(distanceToChannel, 0.52 / minDimension) * 0.30
                      + exp(-distanceToChannel * minDimension / 4.0) * 0.035)
                   * visibility;
        }
    }

    float samples = clamp(sampleCount, 64.0, 1024.0);
    float sampleIndex = floor(clamp(uv.x, 0.0, 0.999999) * samples);
    float x0 = sampleIndex / samples;
    float x1 = (sampleIndex + 1.0) / samples;
    float localX = clamp((uv.x - x0) * samples, 0.0, 1.0);
    float y0 = 0.5 + scopeSample(x0) * amplitude * 0.40;
    float y1 = 0.5 + scopeSample(x1) * amplitude * 0.40;
    float traceY = mix(y0, y1, localX);
    float traceDistance = abs(uv.y - traceY);
    float halfWidth = max(traceWidth * 0.5, 0.85 / minDimension);
    float traceCore = pixelMask(traceDistance, halfWidth);
    float traceRim = max(0.0, pixelMask(traceDistance,
                         halfWidth + 1.25 / minDimension) - traceCore);
    float traceGlow = exp(-max(traceDistance - halfWidth, 0.0)
                          * minDimension / 7.0) * (1.0 - traceCore);
    float energy = 0.90 + uRMS * 0.34 + uOnset * 0.42;
    color += (vec3(1.25) * traceCore
            + traceColor.rgb * (traceCore * 0.46 + traceRim * 1.30
                                + traceGlow * (0.10 + glowAmount * 0.26))) * energy;

    color = vec3(1.0) - exp(-max(color, 0.0) * 1.08);
    gl_FragColor = vec4(pow(color, vec3(0.90)), 1.0);
}
