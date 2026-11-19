// Neon Bass Canyon — a raymarched flight through a procedurally carved
// canyon whose rock walls are striped with glowing equalizer veins. The
// canyon narrows with the low end (bass squeezes the walls apart inward),
// each wall vein pulses with its own spectrum band, and beat-locked lanterns
// drift upward like paper lamps at a festival. Score notes hang as floating
// neon signs over the canyon rim, hue-matched to their pitch. Audio + score +
// clock reactive. Original work for the Arbit Media Machine. License: CC0.

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

// Canyon walls: distance to two carved side slabs + floor.
float mapCanyon(vec3 p)
{
    float fly = uTime * 2.2;
    float z = p.z - fly;

    // Wall profile wanders with layered noise; bass pulls the gap tighter.
    float squeeze = 0.55 * clamp(arbitBand(0.0) * 3.0 + arbitBand(1.0) * 1.5, 0.0, 1.4);
    float wander = vnoise(vec2(z * 0.06, 3.7)) * 1.6 - 0.8;
    float halfGap = max(1.35 - squeeze + wander, 0.45);

    float wallL = -(p.x + halfGap);          // inside: x > -halfGap
    float wallR =   p.x - halfGap;           // inside: x <  halfGap
    float wall = min(wallL, wallR);

    // Rocky surface detail.
    wall -= 0.14 * vnoise(p.xz * 2.3) - 0.07;
    // Floor far below the flight path.
    float floorD = p.y + 1.6 + 0.5 * sin(z * 0.21);
    // Ceiling occasionally opens to sky.
    float ceilD = 3.2 + 1.4 * vnoise(vec2(z * 0.05, 8.2)) - p.y;
    return min(wall, min(floorD, ceilD));
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.003, 0.0);
    return normalize(vec3(
        mapCanyon(p + e.xyy) - mapCanyon(p - e.xyy),
        mapCanyon(p + e.yxy) - mapCanyon(p - e.yxy),
        mapCanyon(p + e.yyx) - mapCanyon(p - e.yyx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // The camera banks gently with the bar phase, riding above the floor.
    vec3 ro = vec3(sin(uBarPhase * 6.2831853) * 0.25,
                   0.15 + 0.1 * sin(uTime * 0.43),
                   0.0);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 96; ++i)
    {
        vec3 p = ro + rd * t;
        float d = mapCanyon(p);
        if (d < 0.0018) { hit = t; break; }
        t += d * 0.9;
        if (t > uCamFar) break;
    }

    // Sky visible where the ceiling opens up.
    vec3 col = mix(vec3(0.05, 0.03, 0.10), vec3(0.16, 0.08, 0.20),
                   smoothstep(-0.1, 0.7, rd.y));
    col += hsv2rgb(vec3(fract(uBeat * 0.01 + 0.75), 0.5, 1.0))
         * pow(max(rd.y, 0.0), 2.0) * 0.10;

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);
        vec3 lig = normalize(vec3(0.2, 0.8, -0.3));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        col = vec3(0.13, 0.11, 0.16) * (0.35 + 0.65 * dif);

        // Equalizer veins: horizontal stripes on the walls, one per band,
        // brightness from that band's energy.
        if (abs(n.x) > 0.55)
        {
            float stripe = fract(p.y * 9.0 - uBeat * 0.4);
            int bandIdx = int(mod(floor(p.y * 9.0 - uBeat * 0.4), 64.0));
            float b = arbitBand(float(bandIdx));
            float vein = smoothstep(0.42, 0.5, stripe) * smoothstep(0.58, 0.5, stripe);
            vec3 hue = hsv2rgb(vec3(fract(0.72 - float(bandIdx) / 64.0 * 0.55), 0.9, 1.0));
            col += hue * vein * b * 4.0;
            col += hue * b * 0.35;   // soft wash even between stripes
        }

        // Beat lanterns drifting up: world-space glow spheres in rows.
        float pulse = 0.6 + 0.8 * uOnset;
        for (int k = 0; k < 5; ++k)
        {
            float seed = float(k) * 17.31;
            float lz = mod(seed + uTime * 0.5, 30.0) - 15.0;
            float lx = (mod(seed, 2.0) * 2.0 - 1.0) * 1.05;
            float ly = mod(seed * 0.37 + uTime * 0.22, 3.5) - 1.4;
            vec3 lp = vec3(lx, ly, lz);
            vec3 w = lp - ro;
            float proj = dot(w, rd);
            if (proj > 0.0)
            {
                float sd = length(w - rd * proj);
                col += vec3(1.0, 0.62, 0.25)
                     * exp(-sd * sd * 320.0) * pulse * 1.8;
            }
        }

        col *= exp(-hit * 0.05);   // depth haze
    }

    // Score signs over the rim: bright marks high on both walls.
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
        float ang = midi * 0.71;
        float sideSign = mod(float(i), 2.0) * 2.0 - 1.0;
        // Signs stream toward the eye with the canyon and wrap, so every
        // sounding note stays ahead of the camera.
        float lane = mod(ang + float(k) * 2.9 - uTime * 2.2, 22.0);
        vec3 sp = vec3(sideSign * 1.28,
                       ro.y + 0.5 + 1.2 * fract(midi / 12.0),
                       ro.z - 2.5 - lane);
        vec3 w = sp - ro;
        float proj = dot(w, rd);
        if (proj > 0.0)
        {
            float sd = length(w - rd * proj);
            vec3 hue = hsv2rgb(vec3(fract(0.66 - tt * 0.5), 0.85, 1.0));
            col += hue * (exp(-sd * sd * 900.0) + exp(-sd * sd * 90.0) * 0.25) * arbitNoteVel(i)
                 * (0.8 + 0.8 * beatPulse(0.4)) * 2.0;
        }
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.78));
    fragColor = vec4(col, 1.0);
}
