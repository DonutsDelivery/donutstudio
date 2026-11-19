// Harmonic Cathedral — a raymarched 3D hall where every sounding note builds
// architecture. Each active note raises a glowing pillar in a ring around the
// camera; the pillar's height follows the note's pitch and its glow breathes
// with the beat. The floor is a reflective lattice that lights up under each
// pillar, and transients kick dust-motes of light through the air. Works with
// no notes too: the hall still breathes with the clock. Audio + score + clock
// reactive. Original work for the Arbit Media Machine. License: CC0.

float hash1(float n){ return fract(sin(n) * 43758.5453); }

// One pillar: a rounded box at angle `ang`, radius `rad`, height driven by
// the caller, once per sounding note.
float sdPillar(vec3 p, float ang, float rad, float hgt)
{
    vec2 c = vec2(cos(ang), sin(ang)) * rad;
    vec3 q = p - vec3(c.x, 0.0, c.y);
    q.y -= hgt * 0.5;
    // vertical capsule-ish column: clamp to its top
    float d = length(max(abs(q.xz) - vec2(0.16), 0.0)) + min(max(q.x, q.z), 0.0);
    return max(d, abs(p.y - hgt * 0.5) - hgt * 0.5);
}

float map(vec3 p)
{
    float d = p.y + 0.02;                       // floor plane just below origin
    float swell = 0.35 * beatPulse(0.5);
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float t = clamp((midi - 24.0) / 72.0, 0.0, 1.0);   // pitch -> height
        float ang = float(k) * 6.2831853 / max(float(sounding), 1.0)
                  + uBeat * 0.15;
        float hgt = 0.7 + 3.2 * t + swell;
        d = min(d, sdPillar(p, ang, 2.6, hgt));
    }
    // A distant ring wall so the hall never reads as empty space.
    float wall = 9.5 - length(p.xz);
    return min(d, max(wall, abs(p.y - 2.0) - 2.0));
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
    vec3 ro, rd;
    arbitCameraRay(fragCoord, ro, rd);

    // Slow ceremonial orbit when idle; holds steady while playing.
    float drift = uPlaying > 0 ? uBeat * 0.05 : uTime * 0.12;
    ro = vec3(sin(drift) * 0.8, 1.05 + 0.06 * sin(uTime * 0.4), cos(drift) * 0.8);
    arbitCameraRay(fragCoord, ro, rd);   // rebuild ray from moved eye

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 90; ++i)
    {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.0015) { hit = t; break; }
        t += d * 0.9;                    // slight relaxation keeps corners clean
        if (t > uCamFar) break;
    }

    // Sky: deep vault with a soft bloom toward the apex.
    vec3 col = mix(vec3(0.010, 0.014, 0.032), vec3(0.05, 0.07, 0.14),
                   smoothstep(-0.2, 0.8, rd.y));
    col += vec3(0.10, 0.08, 0.05) * pow(max(rd.y, 0.0), 3.0);

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);
        float dif = clamp(dot(n, normalize(vec3(0.4, 0.85, 0.25))), 0.0, 1.0);
        float amb = 0.22 + 0.18 * n.y;

        vec3 base = vec3(0.30, 0.34, 0.42);
        col = base * (amb + 0.75 * dif);

        // Light from every sounding pillar: hue by pitch, punch by velocity.
        // Each lamp's pool narrows to the gap between pillars, so a dense
        // chord keeps one distinct pool per note instead of summing to white.
        float pulse = 0.65 + 0.55 * uRMS + 0.9 * uOnset;
        int sounding = arbitSoundingCount();
        float gap = 6.2831853 * 2.6 / max(float(sounding), 1.0);
        float falloff = max(0.55, 4.0 / (gap * gap));
        vec3 lamps = vec3(0.0);
        for (int k = 0; k < sounding; ++k)
        {
            int i = arbitSoundingRow(k);
            float midi = arbitNotePitch(i);
            float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
            float ang = float(k) * 6.2831853 / max(float(sounding), 1.0)
                      + uBeat * 0.15;
            vec2 c = vec2(cos(ang), sin(ang)) * 2.6;
            float dist = length(p.xz - c);
            float lamp = exp(-dist * dist * falloff) * arbitNoteVel(i);
            vec3 hue = hsv2rgb(vec3(fract(0.62 - tt * 0.5), 0.85, 1.0));
            lamps += hue * lamp * pulse * 1.6;

            // Mirror streak on the polished floor.
            if (p.y < 0.08)
                lamps += hue * lamp * 0.8 * (1.0 - smoothstep(0.0, 2.2, hit));
        }
        // Soft tonemap: overlapping pools roll off instead of clipping.
        col += lamps / (1.0 + lamps);
        // Faint grid on the floor for depth cues.
        vec2 g = abs(fract(p.xz * 0.5) - 0.5);
        col += vec3(0.04) * (1.0 - smoothstep(0.0, 0.06, min(g.x, g.y)))
             * step(p.y, 0.01);
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
