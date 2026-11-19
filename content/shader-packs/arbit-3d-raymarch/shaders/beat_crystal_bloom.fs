// Beat Crystal Bloom — a raymarched crystal cluster that grows out of the
// floor on every downbeat and shatters light across the scene. Crystals are
// elongated octahedrons; their count, height, and tilt follow the score (one
// shard per sounding note), the whole cluster refracts a hue by pitch class,
// and sustained notes keep growing while they hold. A reflective obsidian
// floor doubles everything. Audio + score + clock reactive. Original work
// for the Arbit Media Machine. License: CC0.

float hash21(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 443.8975);
    q += dot(q, q.yzx + 19.19);
    return fract((q.x + q.y) * q.z);
}

float sdOctahedron(vec3 p, float s)
{
    p = abs(p);
    return (p.x + p.y + p.z - s) * 0.57735027;
}

// One crystal: an elongated, tilted octahedron at a polar position.
float sdCrystal(vec3 p, float ang, float rad, float hgt, float tilt)
{
    vec2 c = vec2(cos(ang), sin(ang)) * rad;
    vec3 q = p - vec3(c.x, 0.0, c.y);
    // tilt away from center
    q.xz = mat2(cos(tilt), -sin(tilt), sin(tilt), cos(tilt)) * q.xz;
    q.y -= hgt * 0.5;
    return max(sdOctahedron(q, hgt * 0.55), abs(q.y) - hgt * 0.62);
}

float mapScene(vec3 p)
{
    float d = p.y;   // obsidian floor at y=0
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
        // Sustained notes grow: use remaining length as growth budget.
        float sustain = clamp(arbitNoteRemain(i) * 0.25, 0.15, 1.0);
        float ang = midi * 0.9 + uBeat * 0.05;
        float rad = 0.35 + fract(midi / 12.0) * 1.15;
        float hgt = (0.45 + 1.6 * tt) * (0.55 + 0.45 * sustain);
        d = min(d, sdCrystal(p, ang, rad, hgt,
                             0.22 * sin(midi) + uBeatPhase * 0.06));
    }
    // One central spire that breathes with the beat so quiet bars still bloom.
    float swell = 0.7 + 0.5 * beatPulse(0.55) + 0.4 * clamp(uRMS * 2.5, 0.0, 1.2);
    d = min(d, sdCrystal(p, uTime * 0.1, 0.0, swell, 0.0));
    return d;
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.0015, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Low camera looking slightly up: crystals tower over the viewer.
    vec3 ro = vec3(sin(uTime * 0.13) * 0.5,
                   0.35 + 0.08 * sin(uTime * 0.21),
                   cos(uTime * 0.13) * 2.6);
    ro.z += 1.2;   // pull back to frame the central spire
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);
    // The camera helper puts the eye at uCamPos, which by default sits on
    // the floor plane, so every ray would hit the floor at t = 0. Orbit the
    // eye up over the cluster and aim it down at the crystals.
    mat2 lift = mat2(cos(0.28), -sin(0.28), sin(0.28), cos(0.28));
    ro.yz = lift * ro.yz;
    rd.yz = lift * rd.yz;
    ro.y += 0.45;

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 96; ++i)
    {
        vec3 p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.0012) { hit = t; break; }
        t += d;
        if (t > uCamFar) break;
    }

    // Backdrop: dark gradient hall with a slow aurora sheen.
    vec3 col = mix(vec3(0.012, 0.010, 0.028), vec3(0.05, 0.04, 0.10),
                   smoothstep(-0.3, 0.8, rd.y));

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);

        if (p.y < 0.005)
        {
            // Obsidian floor: mirror-dark with a soft reflection of the sky.
            vec3 refl = reflect(rd, vec3(0.0, 1.0, 0.0));
            float fres = pow(1.0 - max(dot(vec3(0.0,1.0,0.0), -rd), 0.0), 2.5);
            col = vec3(0.02) + vec3(0.10, 0.09, 0.16) * fres
                + vec3(0.03, 0.03, 0.06) * smoothstep(-0.1, 0.6, refl.y);
            // Ring highlight where crystals meet glass.
            float ringGlow = exp(-abs(length(p.xz) - 1.0) * 3.0);
            col += vec3(0.30, 0.20, 0.50) * ringGlow * (0.3 + uRMS);
        }
        else
        {
            vec3 lig = normalize(vec3(0.5, 0.85, 0.3));
            float dif = clamp(dot(n, lig), 0.0, 1.0);
            float spec = pow(clamp(dot(reflect(rd, n), lig), 0.0, 1.0), 40.0);

            // Hue by height band (pitch proxy): low = violet, high = cyan.
            float hueBand = fract(0.72 - clamp(p.y / 2.4, 0.0, 1.0) * 0.45
                                  + uBeat * 0.012);
            vec3 base = hsv2rgb(vec3(hueBand, 0.75, 1.0));
            col = base * (0.18 + 0.7 * dif) + vec3(1.0) * spec * 0.9;

            // Inner-light refraction fake: glow strongest inside the shard.
            col += base * exp(-length(cross(normalize(rd), n)) * 0.8)
                 * (0.25 + 1.1 * uRMS + 0.8 * uOnset);
        }
        col *= exp(-hit * 0.045);   // depth haze
    }

    // Sparkle motes around the cluster on transients.
    if (uOnsetAge < 0.5)
    {
        float sparkAmt = (1.0 - uOnsetAge / 0.5);
        vec2 sp = fragCoord / uResolution.xy;
        float gridCell = hash21(floor(sp * vec2(90.0, 50.0))
                                + floor(uTime * 12.0));
        if (gridCell > 0.995) col += vec3(1.0) * sparkAmt * 1.5;
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.80));
    fragColor = vec4(col, 1.0);
}
