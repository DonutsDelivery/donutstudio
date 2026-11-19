// Gravity Well — a raymarched black hole with an accretion disc that is
// literally the mix. Disc brightness follows the 64 spectrum bands mapped
// across its radius (bass at the event horizon, treble at the rim), lensing
// warps the starfield behind it, and every onset slings a hot plasma arm
// around the well. Sounding notes orbit as glowing particles whose speed
// follows their pitch: high notes whip around fast, bass notes crawl.
// Audio + score + clock reactive. Original work for the Arbit Media Machine.
// License: CC0.

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 1031.0);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float hash31(float n){ return fract(sin(n) * 43758.5453); }

// Starfield in a direction (cheap cubemap-free stars).
vec3 stars(vec3 rd)
{
    vec3 col = vec3(0.0);
    vec2 sp = vec2(atan(rd.z, rd.x), asin(clamp(rd.y, -1.0, 1.0)));
    for (int layer = 1; layer <= 2; ++layer)
    {
        float sc = 24.0 * float(layer);
        vec2 cell = floor(sp * sc);
        vec2 f = fract(sp * sc) - 0.5;
        float h = hash12(cell + float(layer) * 7.0);
        if (h > 0.93)
        {
            float tw = 0.6 + 0.4 * sin(uTime * (1.0 + 3.0 * h) + h * 40.0);
            col += vec3(0.8, 0.85, 1.0) * smoothstep(0.25, 0.0, length(f))
                 * (h - 0.9) * 10.0 * tw / float(layer);
        }
    }
    return col;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // The camera slowly circles above the disc plane.
    float orbit = uTime * 0.06;
    vec3 ro = vec3(sin(orbit) * 4.2, 1.15 + 0.2 * sin(uTime * 0.13),
                   cos(orbit) * 4.2);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    // --- Gravitational lensing: bend rays toward the center -------------
    vec3 bh = vec3(0.0);                       // black hole at origin
    float bend = 0.55 + 0.35 * clamp(uRMS * 3.0, 0.0, 1.0);

    // Volumetric disc accumulation along a slightly bent ray.
    float jitter = hash12(fragCoord + fract(uTime) * 57.3);
    float t = 0.4 + jitter * 0.15;
    vec3 acc = vec3(0.0);
    float trans = 1.0;
    bool captured = false;

    vec3 horizonStarDir = rd;
    for (int i = 0; i < 64; ++i)
    {
        vec3 p = ro + rd * t;

        // Pull the sample point toward the hole (fake lensing on position).
        vec3 toC = bh - p;
        float dist = max(length(p), 0.001);
        p += normalize(toC) * bend * 0.22 / (dist * 0.35 + 0.4);

        dist = length(p);
        if (dist < 0.42) { captured = true; break; }   // swallowed

        // Disc: flat slab in xz between inner rim and outer radius.
        float rad = length(p.xz);
        if (abs(p.y) < 0.16 && rad > 0.45 && rad < 3.2)
        {
            // Spectrum bands mapped across the radius, bass inside.
            float bandPos = clamp((rad - 0.45) / 2.75, 0.0, 1.0);
            float b = arbitBand(pow(bandPos, 0.65) * 63.0);
            // Doppler-ish asymmetry: approaching side burns brighter.
            float ang = atan(p.z, p.x);
            float doppler = 0.55 + 0.45 * sin(ang + uTime * 0.8);
            // Onset plasma arm: a bright spiral that tightens after hits.
            float armPhase = ang * 1.5 - log(max(rad, 0.05)) * 3.0
                           + uTime * 1.2 - uOnsetAge * 2.5;
            float arm = exp(-armPhase * armPhase * 6.0)
                      * exp(-uOnsetAge * 1.4) * uOnset * 3.0;
            vec3 hue = mix(vec3(1.0, 0.55, 0.18),
                           hsv2rgb(vec3(fract(0.58 + bandPos * 0.30), 0.85, 1.0)),
                           bandPos);
            float den = b * doppler * (0.5 + 0.5 * hash31(floor(rad * 26.0)))
                      + arm;
            if (den > 0.01)
            {
                float aStep = clamp(den * 0.14, 0.0, 0.5);
                acc += trans * hue * den * 2.6;
                trans *= 1.0 - aStep;
                if (trans < 0.04) break;
            }
        }
        else if (trans > 0.5 && i == 0)
        {
            horizonStarDir = rd;   // remember pre-lensed direction for stars
        }

        // Step grows with distance so far samples stay cheap.
        t += max(0.09 + dist * 0.06, 0.06);
        if (t > 16.0) break;
    }

    // Background stars, dimmed by accumulated disc light; lensing ring adds
    // a faint halo around the shadow.
    vec3 bg = stars(horizonStarDir);
    vec3 col = bg * trans + acc;
    float shadowEdge = smoothstep(0.50, 0.42, length(cross(rd, normalize(bh - ro))));
    col *= mix(1.0, 0.0, shadowEdge * step(0.001, 1.0));
    col += vec3(1.0, 0.72, 0.42) * pow(shadowEdge, 6.0) * (0.4 + 1.6 * uRMS);

    // Score particles orbiting the well.
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
        float speed = 0.4 + 2.2 * tt;                    // high notes fly
        float ang = midi * 1.1 + uTime * speed;
        float orbRad = 0.8 + fract(midi / 17.0) * 1.9;
        vec3 sp = vec3(cos(ang) * orbRad,
                       sin(uBeat * 0.9 + float(i) * 2.1) * 0.12,
                       sin(ang) * orbRad);
        vec3 w = sp - ro;
        float proj = dot(w, rd);
        if (proj > 0.02 && proj < 20.0)
        {
            float sd = length(w - rd * proj);
            bool isRoot = arbitNoteIsRoot(i) > 0.5;
            vec3 hue = isRoot ? vec3(1.0, 0.80, 0.35)
                     : hsv2rgb(vec3(fract(0.60 - tt * 0.48), 0.9, 1.0));
            col += hue * exp(-sd * sd * 1800.0) * arbitNoteVel(i)
                 * (0.9 + 1.4 * beatPulse(0.4)) * 2.2;
        }
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.82));
    fragColor = vec4(col, 1.0);
}
