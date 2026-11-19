// Chromatic Depth Pool — a raymarched pool of liquid color where depth
// itself is the instrument: the surface sits at a height set by the overall
// loudness, waves ripple outward from every sounding note's 2D position at
// its exact pitch, and the liquid's refraction tints deepen with distance —
// low notes stain the deep layers, high notes shimmer at the surface.
// Transient raindrops patter the pool on onsets. Works with no notes as a
// calm breathing pool. Audio + score + clock reactive. Original work for
// the Arbit Media Machine. License: CC0.

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 1031.0);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float vnoise(vec2 p)
{
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash12(i);
    float b = hash12(i + vec2(1.0, 0.0));
    float c = hash12(i + vec2(0.0, 1.0));
    float d = hash12(i + vec2(1.0, 1.0));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

float surfaceHeight(vec2 xz)
{
    // Base breathing level follows loudness; the pool rises when loud.
    float level = -0.25 - 0.35 * clamp(uRMS * 3.0, 0.0, 1.5);

    // Note ripples: each sounding note radiates rings from its spot.
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float cents = arbitNoteCents(i);
        vec2 c = vec2(hash12(vec2(midi, 3.0)) - 0.5,
                      hash12(vec2(cents, 7.0)) - 0.5) * 4.0;
        float freq = pow(2.0, cents / 1200.0) * 3.0;
        float dist = length(xz - c);
        float ringPhase = dist * (2.0 + freq) - uTime * (1.2 + freq * 0.8)
                        - arbitNoteAge(i) * 1.5;
        float fade = exp(-dist * 0.55) * exp(-max(arbitNoteRemain(i), 0.0) * 0.15);
        float amp = 0.05 + 0.16 * arbitNoteVel(i);
        level += sin(ringPhase * 6.2831853) * amp * fade;
    }

    // Onset raindrops: short sharp pips across the whole pool.
    if (uOnsetAge < 0.6)
    {
        float dropGrid = vnoise(xz * 6.0 + floor(uTime * 10.0) * 3.7);
        level -= smoothstep(0.75, 1.0, dropGrid)
               * exp(-uOnsetAge * 5.0) * 0.10 * uOnset;
    }

    // Gentle ambient chop so the surface always lives.
    level += vnoise(xz * 1.8 + uTime * 0.25) * 0.05 - 0.025;
    return level;
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.02, 0.0);
    return normalize(vec3(
        surfaceHeight(p.xz - e.xy) - surfaceHeight(p.xz + e.xy),
        2.0 * e.x,
        surfaceHeight(p.xz - e.yx) - surfaceHeight(p.xz + e.yx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Camera hovers just above the liquid looking down-forward.
    vec3 ro = vec3(sin(uTime * 0.11) * 0.8,
                   1.15 + 0.1 * sin(uTime * 0.31),
                   cos(uTime * 0.11) * 2.4);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    // Floor of the vessel (deep layer).
    float tFloor = (ro.y + 1.6) / max(-rd.y, -0.001);
    bool seesFloor = rd.y < -0.02;

    // March to the wavy surface.
    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 70; ++i)
    {
        vec3 p = ro + rd * t;
        float d = p.y - surfaceHeight(p.xz);
        if (d < 0.004) { hit = t; break; }
        t += max(d * 0.8, 0.03);
        if (t > uCamFar || (seesFloor && t > tFloor)) break;
    }

    // Sky above: soft studio gradient with a cool key light.
    vec3 skyLow = mix(vec3(0.06, 0.08, 0.14), vec3(0.30, 0.38, 0.55),
                      clamp(uRMS * 3.0, 0.0, 1.0));
    vec3 col = mix(skyLow, vec3(0.02, 0.03, 0.07),
                   smoothstep(-0.1, 0.6, rd.y));

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);
        float depth = clamp((p.y + 1.2) / 1.4, 0.0, 1.0);   // 0 deep .. 1 top

        // Absorption tint: deep layers drink red first, so the deep end is
        // teal-indigo and wave crests catch warm highlights.
        vec3 shallow = vec3(0.20, 0.65, 0.70);
        vec3 deepTint = vec3(0.04, 0.09, 0.30);
        vec3 body = mix(deepTint, shallow, pow(depth, 1.4));

        vec3 lig = normalize(vec3(0.35, 0.85, 0.25));
        float fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        float spec = pow(clamp(dot(reflect(rd, n), lig), 0.0, 1.0), 90.0);

        float dif = clamp(dot(n, lig), 0.0, 1.0);
        col = body * (0.45 + 0.55 * dif)
            + skyLow * fres * 0.9
            + vec3(1.0, 0.95, 0.85) * spec * (1.0 + 2.2 * uOnset);

        // See-through hint of the vessel floor far below in calm spots.
        if (seesFloor)
        {
            vec3 fp = ro + rd * tFloor;
            float grid = max(abs(fract(fp.x * 0.7) - 0.5),
                             abs(fract(fp.z * 0.7) - 0.5));
            col += vec3(0.02, 0.05, 0.08)
                 * (1.0 - smoothstep(0.46, 0.5, grid))
                 * (1.0 - depth) * 0.6;
        }
        col *= exp(-hit * 0.03);
    }
    else if (!seesFloor)
    {
        // Ray escapes above the horizon line: pure sky.
        col = mix(skyLow, vec3(0.02, 0.03, 0.07), smoothstep(0.0, 0.6, rd.y));
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.78));
    fragColor = vec4(col, 1.0);
}
