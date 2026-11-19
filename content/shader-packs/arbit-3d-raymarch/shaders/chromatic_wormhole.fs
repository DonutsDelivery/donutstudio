// Chromatic Wormhole — a raymarched flight down an infinite twisted tunnel.
// The tube's surface is a living equalizer: 64 spectrum bands run lengthwise
// as glowing ribs whose amplitude pumps with the music, the tunnel's twist
// tightens with overall loudness, and every transient fires a light wave that
// races ahead of the camera. The walls' hue cycles through the octave so a
// full trip reads as one turn of the chroma wheel. Works with no notes —
// pure clock + audio. Audio + clock reactive. Original work for the Arbit
// Media Machine. License: CC0.

float hash11(float n){ return fract(sin(n * 127.1) * 43758.5453); }

vec2 twist(vec2 p, float amount)
{
    float a = amount;
    float cs = cos(a), sn = sin(a);
    return mat2(cs, -sn, sn, cs) * p;
}

float map(vec3 p)
{
    // Fly forward forever: scroll z by time and beat.
    float fly = uTime * 1.6 + uBeat * 0.35;
    p.z -= fly;

    // Twist grows with loudness; the tunnel breathes on the beat.
    float twistAmt = (0.55 + 1.3 * clamp(uRMS * 2.2 + uPeak, 0.0, 1.5))
                   * p.z * 0.22;
    vec2 q = twist(p.xy, twistAmt);

    // Octagonal tube with band-driven radius modulation.
    float r = length(q);
    float band = clamp(arbitBand(mod(atan(q.y, q.x) / 6.2831853 + 0.5, 1.0) * 63.0), 0.0, 1.0);
    float radius = 1.15
                 - band * 0.45
                 + 0.10 * sin(p.z * 0.7)
                 - 0.06 * beatPulse(0.5);

    // Faceted walls: distance to an octagon-ish shape via max of two planes.
    vec2 o = abs(q);
    float oct = max(dot(o, normalize(vec2(1.0, 0.62))), o.y);
    return oct - radius + 0.06 * hash11(floor(p.z * 3.0)) * band;
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.002, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Camera rides the axis with a lazy figure-eight wobble.
    vec3 ro = vec3(0.18 * sin(uTime * 0.5),
                   0.18 * sin(uTime * 0.37 + 1.7),
                   0.0);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 90; ++i)
    {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.0015) { hit = t; break; }
        t += d * 0.95;
        if (t > 24.0) break;
    }

    vec3 col = vec3(0.006, 0.004, 0.010);
    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        float fly = uTime * 1.6 + uBeat * 0.35;
        float twistAmt = (0.55 + 1.3 * clamp(uRMS * 2.2 + uPeak, 0.0, 1.5))
                       * (p.z - fly) * 0.22;
        vec2 pq = twist(p.xy, twistAmt);
        float ang = atan(pq.y, pq.x) / 6.2831853 + 0.5;

        vec3 n = calcNormal(p);
        float dif = clamp(dot(n, -rd + vec3(0.0, 0.0, 0.4)), 0.0, 1.0);

        // Ribs: each angle slice is one spectrum band, lit by its energy.
        float bandPos = fract(ang * 63.0 / 63.0);
        float b = clamp(arbitBand(bandPos * 63.0) * 3.2, 0.0, 1.0);
        float ribLine = smoothstep(0.30, 0.34, abs(fract(ang * 32.0) - 0.5));
        vec3 hue = hsv2rgb(vec3(fract(ang + uBeat * 0.03), 0.85, 1.0));
        col = hue * (0.10 + 1.5 * dif)
            + hue * b * (0.8 + ribLine * 1.4)
            + vec3(1.0) * b * ribLine * 0.35;

        // Transient shockwave racing forward: a bright ring in world z.
        if (uOnsetAge < 1.2)
        {
            float wz = mod(p.z - fly, 26.0);
            float ring = exp(-abs(wz - uOnsetAge * 14.0) * 1.8);
            col += vec3(0.9, 0.95, 1.0) * ring * (1.0 - uOnsetAge / 1.2) * 2.0;
        }

        // Distance fade into the dark end of the hole.
        col *= exp(-hit * 0.09);
    }

    // Speed-line streaks toward the vanishing point when playing hard.
    float streak = pow(max(0.0, 1.0 - length(rd.xy) * 2.2), 6.0);
    col += hsv2rgb(vec3(fract(uBeat * 0.05), 0.6, 1.0)) * streak * uRMS * 0.9;

    col = pow(clamp(col, 0.0, 1.0), vec3(0.80));
    fragColor = vec4(col, 1.0);
}
