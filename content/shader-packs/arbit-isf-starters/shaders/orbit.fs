/*{
  "DESCRIPTION": "Orbit — project notes become luminous bodies whose pitch, velocity, root role, exact tuning and harmonic links shape a beat-locked orbital system. With no score, spectrum bands drive the editable fallback bodies. center/count/size/spread/color remain ordinary ISF parameters. Original work for DonutStudio. License: CC0.",
  "CREDIT": "DonutStudio",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Score", "Audio Visualizer"],
  "INPUTS": [
    { "NAME": "center", "TYPE": "point2D", "DEFAULT": [0.5, 0.5] },
    { "NAME": "count",  "TYPE": "long",  "DEFAULT": 8, "MIN": 1, "MAX": 12 },
    { "NAME": "size",   "TYPE": "float", "DEFAULT": 0.032, "MIN": 0.005, "MAX": 0.2 },
    { "NAME": "spread", "TYPE": "float", "DEFAULT": 0.3, "MIN": 0.05, "MAX": 0.7 },
    { "NAME": "color",  "TYPE": "color", "DEFAULT": [0.18, 0.68, 1.0, 1.0] }
  ]
}*/

#define TAU 6.28318530718

float segmentDistance(vec2 point, vec2 start, vec2 end)
{
    vec2 line = end - start;
    float amount = clamp(dot(point - start, line) / max(dot(line, line), 1e-6), 0.0, 1.0);
    return length(point - start - line * amount);
}

float pixelMask(float distanceValue, float radius)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(radius - aa, radius + aa, distanceValue);
}

float exactMidi(ArbitNote note)
{
    float frequency = max(note.t1.x, 1.0);
    return 69.0 + 12.0 * log2(frequency / 440.0);
}

float noteVisibility(ArbitNote note)
{
    return smoothstep(-0.18, 0.08, note.t0.z)
         * smoothstep(-0.35, 0.12, note.t0.w);
}

vec2 scoreOrbitPosition(ArbitNote note, float index, vec2 origin, float aspect)
{
    float midi = exactMidi(note);
    float pitchClass = fract(midi / 12.0);
    float primeShape = note.t2.z * 0.10 + note.t2.w * 0.055 + note.t3.x * 0.035;
    float radius = spread * (0.48 + 0.34 * fract(midi * 0.6180339) + abs(primeShape));
    float speed = 0.16 + 0.025 * fract(midi / 7.0);
    float phase = pitchClass * TAU + uBeat * speed + index * 0.31 + primeShape;
    vec2 offset = vec2(cos(phase), sin(phase * (1.0 + 0.08 * note.t2.y))) * radius;
    offset.x /= aspect;
    return origin + offset;
}

vec2 fallbackOrbitPosition(float index, float bodies, vec2 origin, float aspect)
{
    float bandPosition = (index + 0.5) / max(bodies, 1.0);
    float bandEnergy = clamp(arbitBand(bandPosition * 63.0) * 4.0, 0.0, 1.0);
    float phase = uBeat * (0.20 + index * 0.015) + index * TAU / max(bodies, 1.0);
    float radius = spread * (0.55 + bandEnergy * 0.55);
    vec2 offset = vec2(cos(phase), sin(phase * (1.25 + index * 0.025))) * radius;
    offset.x /= aspect;
    return origin + offset;
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec2 origin = center;
    vec3 col = mix(vec3(0.004, 0.007, 0.018), vec3(0.010, 0.018, 0.042), uv.y);

    float centreDistance = distance(uv, origin);
    float atmosphere = exp(-centreDistance * 3.2);
    col += mix(vec3(0.035, 0.018, 0.055), vec3(0.095, 0.060, 0.020),
               0.35 + 0.35 * uRMS) * atmosphere;
    float beatRingDistance = abs(centreDistance - spread * (0.36 + 0.08 * beatPulse(0.2)));
    col += color.rgb * exp(-beatRingDistance * minDimension / 5.5) * 0.085;

    bool scoreDriven = uNoteCount > 0;
    int bodyLimit = max(int(count), 1);
    if (scoreDriven)
    {
        for (int edge = 0; edge < uLinkCount; ++edge)
        {
            ArbitLink link = arbitLink(edge);
            int slaveIndex = int(link.t0.x + 0.5);
            int masterIndex = int(link.t0.y + 0.5);
            if (slaveIndex < 0 || masterIndex < 0
                || slaveIndex >= uNoteCount || masterIndex >= uNoteCount) continue;
            ArbitNote slave = arbitNote(slaveIndex);
            ArbitNote master = arbitNote(masterIndex);
            float visibility = min(noteVisibility(slave), noteVisibility(master));
            if (visibility <= 0.001) continue;
            vec2 start = scoreOrbitPosition(slave, float(slaveIndex), origin, aspect);
            vec2 end = scoreOrbitPosition(master, float(masterIndex), origin, aspect);
            float distanceToLink = segmentDistance(uv, start, end);
            float linkCore = pixelMask(distanceToLink, 0.75 / minDimension);
            float linkHalo = exp(-distanceToLink * minDimension / 4.5) * (1.0 - linkCore);
            float ratioHue = fract(log2(max(link.t0.z / max(link.t0.w, 1.0), 0.001)) * 0.28);
            vec3 linkColor = hsv2rgb(vec3(ratioHue, 0.82, 1.0));
            col += linkColor * (linkCore * 1.05 + linkHalo * 0.34) * visibility;
        }
    }

    // Every resident note gets a body; count only sizes the fallback.
    int bodies = scoreDriven ? uNoteCount : bodyLimit;
    for (int i = 0; i < bodies; ++i)
    {
        float index = float(i);
        float visibility = 1.0;
        float velocity = 0.65;
        float root = 0.0;
        float midi = index * 7.0 + 36.0;
        vec2 position;
        if (scoreDriven)
        {
            ArbitNote note = arbitNote(i);
            visibility = noteVisibility(note);
            if (visibility <= 0.001) continue;
            velocity = clamp(note.t0.y, 0.0, 1.0);
            root = clamp(note.t1.w, 0.0, 1.0);
            midi = exactMidi(note);
            position = scoreOrbitPosition(note, index, origin, aspect);
        }
        else
        {
            position = fallbackOrbitPosition(index, float(bodyLimit), origin, aspect);
            velocity = clamp(arbitBand((index + 0.5) / float(bodyLimit) * 63.0) * 4.0,
                             0.15, 1.0);
        }

        float radius = size * (0.18 + velocity * 0.12 + root * 0.08)
                     * (1.0 + uOnset * 0.18);
        float distanceToBody = distance(uv, position);
        float core = pixelMask(distanceToBody, radius);
        float rim = max(0.0, pixelMask(distanceToBody,
                        radius + 1.5 / minDimension) - core);
        float outsidePixels = max(distanceToBody - radius, 0.0) * minDimension;
        float innerGlow = exp(-outsidePixels / 8.5) * (1.0 - core);
        float wideGlow = exp(-outsidePixels / 24.0) * (1.0 - core);
        float specular = pixelMask(distanceToBody, radius * 0.26);
        float hue = fract(midi / 24.0 + uBeat * 0.012 + index * 0.065);
        vec3 pitchColor = hsv2rgb(vec3(hue, 0.88, 1.0));
        vec3 bodyColor = mix(color.rgb, pitchColor, scoreDriven ? 0.78 : 0.58);
        bodyColor = mix(bodyColor, vec3(1.0, 0.72, 0.16), root);
        float energy = visibility * (0.68 + velocity * 0.88 + uRMS * 0.72 + uOnset * 0.68);
        col += (bodyColor * (core * 2.15 + rim * 2.4 + innerGlow * 0.82 + wideGlow * 0.22)
              + vec3(1.0, 0.94, 0.82) * specular * 0.82) * energy;
    }

    col = vec3(1.0) - exp(-max(col, 0.0) * 1.15);
    gl_FragColor = vec4(pow(col, vec3(0.88)), 1.0);
}
