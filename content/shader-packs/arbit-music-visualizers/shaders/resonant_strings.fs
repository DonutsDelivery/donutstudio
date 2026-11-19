/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "Ratio strings with visible wavelengths. The root spans four spatial cycles by default, so 5/4, 3/2, and 7/4 notes span five, six, and seven cycles while frame-timed trails show rapid vibration.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Score", "Just Intonation", "Standing Waves", "Generator"],
  "INPUTS": [
    { "NAME": "amplitude", "TYPE": "float", "DEFAULT": 0.085, "MIN": 0.01, "MAX": 0.18, "LABEL": "Vibration amplitude" },
    { "NAME": "spatialCycles", "TYPE": "float", "DEFAULT": 4.0, "MIN": 1.0, "MAX": 12.0, "LABEL": "Root spatial cycles" },
    { "NAME": "vibrationRate", "TYPE": "float", "DEFAULT": 5.7, "MIN": 0.2, "MAX": 20.0, "LABEL": "Vibrations per beat" },
    { "NAME": "motionTrails", "TYPE": "float", "DEFAULT": 0.68, "MIN": 0.0, "MAX": 1.0, "LABEL": "Motion trails" },
    { "NAME": "lineWidth", "TYPE": "float", "DEFAULT": 0.0028, "MIN": 0.001, "MAX": 0.02, "LABEL": "String width" },
    { "NAME": "harmonicDetail", "TYPE": "float", "DEFAULT": 0.48, "MIN": 0.0, "MAX": 1.0, "LABEL": "Upper modes" },
    { "NAME": "verticalSpread", "TYPE": "float", "DEFAULT": 0.82, "MIN": 0.30, "MAX": 0.92, "LABEL": "Pitch spread" },
    { "NAME": "stringColor", "TYPE": "color", "DEFAULT": [0.12, 0.78, 1.0, 1.0], "LABEL": "String color" },
    { "NAME": "linkColor", "TYPE": "color", "DEFAULT": [0.72, 0.28, 1.0, 1.0], "LABEL": "Ratio links" },
    { "NAME": "rootColor", "TYPE": "color", "DEFAULT": [1.0, 0.70, 0.12, 1.0], "LABEL": "Root string" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.003, 0.006, 0.016, 1.0], "LABEL": "Background" }
  ]
}*/

#define PI 3.14159265359
#define TAU 6.28318530718

float segmentDistance(vec2 point, vec2 start, vec2 end)
{
    vec2 segment = end - start;
    float amount = clamp(dot(point - start, segment) / max(dot(segment, segment), 1e-6), 0.0, 1.0);
    return length(point - start - segment * amount);
}

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

float stringBaseForNote(ArbitNote note, int row)
{
    float spread = clamp(verticalSpread, 0.1, 0.95);
    float bottom = 0.5 - spread * 0.5;
    float exactLogRatio = log2(noteRatio(note));
    float pitchClassLane = fract(exactLogRatio);
    float octaveNudge = clamp(floor(exactLogRatio), -4.0, 4.0) * 0.006;
    float duplicateNudge = (fract((float(row) + 1.0) * 0.61803398875) - 0.5) * 0.016;
    return bottom + clamp(pitchClassLane + octaveNudge + duplicateNudge, 0.015, 0.985) * spread;
}

vec3 standingModes(float x, float ratio)
{
    float carrierCycles = spatialCycles * ratio;
    float carrier = TAU * x * carrierCycles;
    float endpointEnvelope = sin(PI * clamp(x, 0.0, 1.0));
    return vec3(sin(carrier), sin(carrier * 2.0), sin(carrier * 3.0))
         * endpointEnvelope;
}

float standingWave(vec3 modes, float phase)
{
    float sine = sin(phase);
    float cosine = cos(phase);
    float sine2 = 2.0 * sine * cosine;
    float sine3 = sine * (3.0 - 4.0 * sine * sine);
    return modes.x * sine
         + harmonicDetail * modes.y * sine2 * 0.30
         + harmonicDetail * harmonicDetail * modes.z * sine3 * 0.14;
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec3 color = backgroundColor.rgb;

    float resonance = exp(-abs(uv.x - 0.5) * 3.0) * (0.045 + uRMS * 0.075);
    color += mix(vec3(0.025, 0.016, 0.065), vec3(0.014, 0.060, 0.075), uv.y) * resonance;

    for (int edge = 0; edge < uLinkCount; ++edge)
    {
        ArbitLink link = arbitLink(edge);
        int slaveIndex = int(link.t0.x + 0.5);
        int masterIndex = int(link.t0.y + 0.5);
        if (slaveIndex < 0 || masterIndex < 0
            || slaveIndex >= uNoteCount || masterIndex >= uNoteCount) continue;
        float visibility = min(arbitNoteActive(slaveIndex), arbitNoteActive(masterIndex));
        if (visibility <= 0.001) continue;
        float slaveY = stringBaseForNote(arbitNote(slaveIndex), slaveIndex);
        float masterY = stringBaseForNote(arbitNote(masterIndex), masterIndex);
        float ratio = max(link.t0.z / max(link.t0.w, 1.0), 0.001);
        float bridgeX = 0.16 + fract(log2(ratio) * 0.72 + 0.5) * 0.68;
        float bridgeDistance = segmentDistance(uv, vec2(bridgeX, slaveY), vec2(bridgeX, masterY));
        float bridgeCore = pixelMask(bridgeDistance, 0.62 / minDimension);
        float bridgeGlow = exp(-bridgeDistance * minDimension / 5.0) * (1.0 - bridgeCore);
        vec3 ratioColor = mix(linkColor.rgb,
            hsv2rgb(vec3(fract(log2(ratio) * 0.30 + 0.62), 0.80, 1.0)), 0.52);
        color += ratioColor * (bridgeCore * 0.68 + bridgeGlow * 0.17) * visibility;
        float nodeDistanceA = distance(uv, vec2(bridgeX, slaveY));
        float nodeDistanceB = distance(uv, vec2(bridgeX, masterY));
        color += ratioColor * (pixelMask(nodeDistanceA, 2.0 / minDimension)
                             + pixelMask(nodeDistanceB, 2.0 / minDimension))
               * visibility * 0.72;
    }

    for (int i = 0; i < uNoteCount; ++i)
    {
        ArbitNote note = arbitNote(i);
        float visibility = arbitNoteActive(i);
        if (visibility <= 0.001) continue;
        float velocity = clamp(note.t0.y, 0.0, 1.0);
        float ratio = noteRatio(note);
        float baseY = stringBaseForNote(note, i);
        float phaseCycles = uBeat * vibrationRate * ratio;
        float phase = TAU * phaseCycles;
        float frameSeconds = clamp(TIMEDELTA, 1.0 / 240.0, 1.0 / 12.0);
        float framePhaseStep = TAU * vibrationRate * ratio
                             * (max(uBPM, 1.0) / 60.0) * frameSeconds;
        float displacementScale = amplitude * (0.32 + velocity * 0.68)
                                * visibility * (1.0 + uOnset * 0.18);
        vec3 modes = standingModes(uv.x, ratio);
        float currentY = baseY + standingWave(modes, phase) * displacementScale;
        float trailY1 = baseY + standingWave(modes, phase - framePhaseStep) * displacementScale;
        float trailY2 = baseY + standingWave(modes, phase - framePhaseStep * 2.0) * displacementScale;
        float trailY3 = baseY + standingWave(modes, phase + framePhaseStep) * displacementScale;
        float trailY4 = baseY + standingWave(modes, phase + framePhaseStep * 2.0) * displacementScale;
        float halfWidth = max(lineWidth * 0.5, 0.85 / minDimension);

        vec3 ratioColor = hsv2rgb(vec3(fract(log2(ratio) * 0.34 + 0.54), 0.82, 1.0));
        vec3 noteColor = mix(stringColor.rgb, ratioColor, 0.62);
        noteColor = mix(noteColor, rootColor.rgb, clamp(note.t1.w, 0.0, 1.0));

        float trailDistance1 = abs(uv.y - trailY1);
        float trailDistance2 = abs(uv.y - trailY2);
        float trailDistance3 = abs(uv.y - trailY3);
        float trailDistance4 = abs(uv.y - trailY4);
        float trail1 = pixelMask(trailDistance1, halfWidth * 0.74);
        float trail2 = pixelMask(trailDistance2, halfWidth * 0.62);
        float trail3 = pixelMask(trailDistance3, halfWidth * 0.74);
        float trail4 = pixelMask(trailDistance4, halfWidth * 0.62);
        color += noteColor * ((trail1 + trail3) * 0.19 + (trail2 + trail4) * 0.085)
               * motionTrails * visibility;

        float distanceToString = abs(uv.y - currentY);
        float core = pixelMask(distanceToString, halfWidth);
        float rim = max(0.0, pixelMask(distanceToString,
                        halfWidth + 1.15 / minDimension) - core);
        float glow = exp(-max(distanceToString - halfWidth, 0.0)
                         * minDimension / 6.0) * (1.0 - core);
        float energy = 0.72 + velocity * 0.62 + uRMS * 0.34 + uOnset * 0.44;
        color += (vec3(1.14) * core
                + noteColor * (core * 0.58 + rim * 1.42 + glow * 0.20))
               * visibility * energy;

        float guideDistance = abs(uv.y - baseY);
        color += noteColor * pixelMask(guideDistance, 0.42 / minDimension)
               * visibility * 0.055;
        float endpointA = distance(uv, vec2(0.0, baseY));
        float endpointB = distance(uv, vec2(1.0, baseY));
        color += noteColor * (pixelMask(endpointA, 2.35 / minDimension)
                            + pixelMask(endpointB, 2.35 / minDimension))
               * visibility * 0.88;
    }

    color = vec3(1.0) - exp(-max(color, 0.0) * 1.12);
    gl_FragColor = vec4(pow(color, vec3(0.88)), 1.0);
}
