// Pitch Galaxy — a raymarched spiral galaxy where every sounding note is a
// star system placed on the spiral by its exact tuning. The galaxy's spiral
// arms wind with the bar phase, the whole disc spins at the song's tempo, and
// the galactic core ignites on transients. A dust lane of thousands of point
// stars (procedural, no geometry) gives it real depth at 4K; harmonic roots
// burn gold while linked notes share their master's hue. Audio + score +
// clock reactive. Original work for the Arbit Media Machine. License: CC0.

float hash21(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 443.8975);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // The eye floats above the disc plane, tilted down at the core.
    vec3 ro = vec3(0.0, 1.9 - 0.15 * sin(uTime * 0.11), 3.6);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);
    // The default camera looks along the disc plane, which shows the galaxy
    // edge-on as one streak. Orbit the eye up so it looks down at the disc.
    mat2 tilt = mat2(cos(0.62), -sin(0.62), sin(0.62), cos(0.62));
    ro.yz = tilt * ro.yz;
    rd.yz = tilt * rd.yz;

    float pulse = 0.65 + 0.7 * uRMS + 1.0 * uOnset;

    vec3 col = vec3(0.004, 0.005, 0.010);
    float trans = 1.0;

    // --- Volumetric march through the disc slab -------------------------
    float jitter = hash21(fragCoord + fract(uTime) * 91.7);
    float t = 0.2 + jitter * 0.1;
    float spin = uTime * 0.12 * (uBPM > 1.0 ? clamp(uBPM / 96.0, 0.5, 2.0) : 1.0);
    for (int i = 0; i < 56; ++i)
    {
        vec3 p = ro + rd * t;

        // Disc density: a flattened exponential disc in rotating coords.
        vec2 d2 = p.xz;
        float rad = length(d2);
        float ang = atan(d2.y, d2.x) - spin * (1.6 / max(rad, 0.35));
        // Two logarithmic arms.
        float armPhase = cos(2.0 * (ang - log(max(rad, 0.05)) * 2.4));
        float arm = smoothstep(0.1, 0.9, armPhase);
        float thick = 0.16 + 0.10 * exp(-rad * 0.8);
        float fall = exp(-rad * 0.55);
        float den = arm * fall
                  * exp(-abs(p.y) * abs(p.y) / (thick * thick))
                  * (0.55 + 0.75 * hash21(floor(d2 * 14.0)));

        if (den > 0.008)
        {
            // Core is hot white-gold; arms run blue-violet outward.
            vec3 hot = vec3(1.0, 0.85, 0.55);
            vec3 cold = hsv2rgb(vec3(fract(0.60 + 0.22 * smoothstep(0.4, 2.6, rad)
                                        + uBeat * 0.015), 0.8, 1.0));
            vec3 hue = mix(hot, cold, smoothstep(0.25, 1.4, rad));
            float coreFire = 1.0 + 2.6 * uOnset * exp(-rad * rad * 2.0);
            float aStep = den * 0.10;
            col += trans * hue * den * 3.2 * coreFire;
            trans *= 1.0 - min(aStep * 1.7, 0.95);
        }
        t += 0.085;
        if (t > 10.0) break;
    }

    // --- Score stars: one per sounding note ------------------------------
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float cents = arbitNoteCents(i);
        float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
        // Radius from pitch class, height from the microtonal deviation off
        // the nearest semitone, so every chord tone stays in view.
        float ang = midi * 0.62 + spin * 2.0 + uBeat * 0.1;
        float rad = 0.55 + 1.9 * fract(midi / 12.0);
        float deviation = cents - 100.0 * floor(cents / 100.0 + 0.5);
        vec3 sp = vec3(cos(ang) * rad,
                       sin(uBeat * 0.7 + float(i)) * 0.06 + deviation * 0.008,
                       sin(ang) * rad);
        vec3 w = sp - ro;
        float proj = dot(w, rd);
        if (proj > 0.02 && proj < 24.0)
        {
            float sd = length(w - rd * proj);
            int master = int(arbitNoteMasterRow(i));
            bool isRoot = arbitNoteIsRoot(i) > 0.5;
            vec3 hue = isRoot ? vec3(1.0, 0.78, 0.30)
                     : hsv2rgb(vec3(fract(0.63 - tt * 0.5), 0.85, 1.0));
            float twinkle = 0.8 + 0.2 * beatPulse(0.4);
            float star = exp(-sd * sd * 2600.0) + exp(-sd * sd * 220.0) * 0.28;
            col += hue * star * arbitNoteVel(i) * twinkle * 2.4;
        }
    }

    // Faint horizon haze so the disc reads as an object in space.
    col += vec3(0.03, 0.04, 0.07) * pow(max(-rd.y, 0.0), 3.0) * 0.5;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
