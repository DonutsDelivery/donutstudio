// Bass Nebula — a raymarched volumetric nebula that lives in the low end.
// The camera drifts through layered fractal clouds whose density is pumped by
// the bass bands; every transient detonates a shockwave that ripples outward
// through the gas. A young star cluster forms from the score: each sounding
// note is a star, positioned by its exact cents offset from the song root, so
// a just-intonation chord paints a tight constellation and a wild tuning
// scatters it. Audio + score + clock reactive. Original work for the Arbit
// Media Machine. License: CC0.

float hash1(vec3 p)
{
    p = fract(p * 0.3183099 + vec3(0.1, 0.2, 0.3));
    p *= 17.0;
    return fract(p.x * p.y * p.z * (p.x + p.y + p.z));
}

float vnoise(vec3 x)
{
    vec3 i = floor(x);
    vec3 f = fract(x);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(mix(hash1(i + vec3(0,0,0)), hash1(i + vec3(1,0,0)), f.x),
                   mix(hash1(i + vec3(0,1,0)), hash1(i + vec3(1,1,0)), f.x), f.y),
               mix(mix(hash1(i + vec3(0,0,1)), hash1(i + vec3(1,0,1)), f.x),
                   mix(hash1(i + vec3(0,1,1)), hash1(i + vec3(1,1,1)), f.x), f.y),
               f.z);
}

float fbm(vec3 p)
{
    float a = 0.5, s = 0.0;
    for (int i = 0; i < 4; ++i) { s += a * vnoise(p); p *= 2.03; a *= 0.5; }
    return s;
}

// Cloud density at a world point. Bands 0..7 are the sub-bass / low end.
float density(vec3 p)
{
    float bass = clamp(arbitBand(0.0) * 1.4 + arbitBand(2.0) * 1.1, 0.0, 1.6);
    vec3 q = p * 0.55 + vec3(0.0, uTime * 0.06, uTime * 0.03);
    float n = fbm(q);
    // Shockwave rings from recent transients (age-faded).
    float ring = 0.0;
    if (uOnsetAge < 1.5)
    {
        float r = length(p) - uOnsetAge * 3.5;
        ring = exp(-r * r * 2.5) * (1.0 - uOnsetAge / 1.5);
    }
    float shell = smoothstep(1.2, 3.8, length(p));   // keep a walkable core
    return max(n * (0.35 + 1.25 * bass) + ring * 0.8 - 0.28, 0.0) * shell;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Drift the eye slowly forward and sideways through the cloud.
    vec3 ro = vec3(sin(uTime * 0.07) * 1.2,
                   cos(uTime * 0.05) * 0.5,
                   uTime * 0.45);
    vec3 ro2 = ro;
    vec3 rd;
    arbitCameraRay(fragCoord, ro2, rd);

    // Volumetric march: 48 steps, jittered to hide banding.
    float jitter = hash1(vec3(fragCoord, uFrame)) ;
    float t = 0.35 + jitter * 0.12;
    vec3 acc = vec3(0.0);
    float trans = 1.0;
    for (int i = 0; i < 48; ++i)
    {
        vec3 p = ro2 + rd * t;
        float den = density(p);
        if (den > 0.01)
        {
            vec3 hue = hsv2rgb(vec3(fract(0.62 - length(p) * 0.045
                                        + uBeat * 0.02), 0.7, 1.0));
            float glowAmp = 0.16 + 0.85 * uRMS;
            vec3 emit = hue * den * glowAmp;
            float aStep = den * 0.09;
            acc += trans * emit * aStep;
            trans *= 1.0 - min(aStep * 1.6, 0.95);
            if (trans < 0.03) break;
        }
        t += 0.14;
        if (t > 11.0) break;
    }

    // Deep-space backdrop behind everything.
    vec3 col = acc + vec3(0.004, 0.005, 0.012) * (1.0 - trans);

    // Star cluster in the clear core: one star per sounding note. The angle
    // is the note's pitch class in exact cents from the song root, and the
    // radius grows with the octave, so a just chord packs tight and a wild
    // tuning scatters. Gas in front dims a star but never hides it.
    int sounding = arbitSoundingCount();
    float twinkle = 0.75 + 0.25 * beatPulse(0.35);
    for (int k = 0; k < sounding; ++k)
    {
        int j = arbitSoundingRow(k);
        float cents = arbitNoteCents(j);
        float octave = floor(cents / 1200.0);
        float ang = (cents - octave * 1200.0) / 1200.0 * 6.2831853;
        float rad = 0.3 + 0.16 * clamp(octave + 3.0, 0.0, 6.0);
        vec3 sp = vec3(cos(ang) * rad, sin(ang) * rad, 0.15 * sin(cents * 0.013));
        vec3 w = sp - ro2;
        float proj = dot(w, rd);
        if (proj <= 0.0) continue;
        float sd = length(w - rd * proj);
        float star = exp(-sd * sd * 3200.0) + exp(-sd * sd * 260.0) * 0.3;
        col += vec3(1.0, 0.92, 0.78) * star * arbitNoteVel(j) * twinkle
             * (0.45 + 0.55 * trans) * 2.2;
    }
    col += vec3(0.05, 0.07, 0.12) * pow(max(rd.y, 0.0), 2.0) * 0.4;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.85));
    fragColor = vec4(col, 1.0);
}
