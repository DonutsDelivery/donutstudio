// Resonance Ocean — a raymarched ocean of standing waves where the sea
// surface is itself a JI chord. Every sounding note drives one wave train at
// its exact root-relative frequency ratio, so a 3:2 makes wide slow swells
// and a 15:16 makes tight fast ripples; dissonant chords literally roughen
// the water. The camera skims just above the surface, sun glitter follows
// transients, and the sky burns with the low end. Audio + score + clock
// reactive. Original work for the Arbit Media Machine. License: CC0.

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 1031.0);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float waveField(vec2 xz)
{
    float h = 0.0;
    // Score-driven trains, one per sounding note: frequency = ratio * base, so exact
    // just-intonation ratios create physically locked interference. Trains
    // add with unrelated phases, so their sum grows with the square root of
    // the count: dividing by it keeps a dense chord as calm as a triad. Each
    // train's height also falls with its frequency, so high notes ripple
    // the surface without tearing it.
    int sounding = arbitSoundingCount();
    float perNote = 0.22 / sqrt(max(float(sounding) / 3.0, 1.0));
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float cents = arbitNoteCents(i);
        float freq = pow(2.0, cents / 1200.0) * 2.2;
        float phase = uTime * freq * 1.7 + cents * 0.01;
        h += sin(dot(xz, vec2(cos(phase * 0.13), sin(phase * 0.13))) * freq + phase)
           * arbitNoteVel(i) * perNote * min(2.2 / freq, 1.0);
    }
    // Always-present swell so it never reads as glass.
    h += sin(xz.x * 0.7 + uTime * 0.8) * cos(xz.y * 0.6 - uTime * 0.6) * 0.10;
    return h;
}

float mapWater(vec3 p)
{
    return p.y - waveField(p.xz);
}

vec3 calcNormalW(vec3 p)
{
    vec2 e = vec2(0.02, 0.0);
    float hC = waveField(p.xz);
    return normalize(vec3(
        hC - waveField(p.xz + e.xy),
        e.x,
        hC - waveField(p.xz + e.yx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Skim low across the surface, gently banking with the bar.
    float barSway = sin(uBarPhase * 6.2831853);
    vec3 ro = vec3(barSway * 0.35,
                   0.55 + 0.08 * sin(uTime * 0.5) + 0.5 * clamp(uRMS * 2.0, 0.0, 1.0),
                   uTime * 0.9);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);
    // The camera helper puts the eye at uCamPos, which by default sits on
    // the mean sea level. Lift it to skim above the swell.
    ro.y += 0.55;

    // March the heightfield.
    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 80; ++i)
    {
        vec3 p = ro + rd * t;
        float d = mapWater(p);
        if (d < 0.004) { hit = t; break; }
        t += max(d * 0.85, 0.02);
        if (t > uCamFar) break;
    }

    // Sky: horizon fire driven by the bass bands, deep zenith above.
    float bass = clamp(arbitBand(0.0) * 2.4 + arbitBand(1.0) * 1.6, 0.0, 1.0);
    vec3 skyLow = mix(vec3(0.10, 0.10, 0.18), vec3(0.85, 0.35, 0.10), bass);
    vec3 col = mix(skyLow * smoothstep(-0.05, 0.25, rd.y),
                   vec3(0.02, 0.03, 0.09), smoothstep(0.0, 0.7, rd.y));
    // Sun disc.
    vec3 sunDir = normalize(vec3(0.3, 0.28, -0.9));
    col += vec3(1.0, 0.75, 0.45) * pow(max(dot(rd, sunDir), 0.0), 220.0) * 2.2;

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormalW(p);
        float fres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
        // Reflected sky + refracted deep teal.
        vec3 refl = mix(skyLow, vec3(0.02, 0.04, 0.10), 0.4);
        vec3 deep = vec3(0.01, 0.06, 0.09) * (0.6 + 0.8 * uRMS);
        col = mix(deep, refl, clamp(fres * 1.4, 0.0, 1.0));

        // Sun glitter path.
        float spec = pow(max(dot(reflect(rd, n), sunDir), 0.0), 60.0);
        col += vec3(1.0, 0.72, 0.40) * spec * (1.2 + 2.5 * uOnset);

        // Foam on steep interference peaks: where notes clash, the sea whitens.
        float energy = abs(waveField(p.xz)) ;
        float foam = smoothstep(0.34, 0.5, energy);
        col += vec3(0.9) * foam * 0.35;

        // Distance haze into the horizon color.
        col = mix(col, skyLow, 1.0 - exp(-hit * 0.045));
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.70));
    fragColor = vec4(col, 1.0);
}
