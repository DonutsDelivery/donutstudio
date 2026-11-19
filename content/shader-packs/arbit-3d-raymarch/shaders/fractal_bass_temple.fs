// Fractal Bass Temple — an infinite raymarched Menger-style temple that
// folds open on the beat. The fractal's hole pattern is carved deeper as the
// low bands hit, every transient fires a light pulse up the central shaft,
// and the whole structure slowly rotates around the camera. A ring of score
// runes (one per sounding note) orbits at the temple's mid height, each rune
// colored by its pitch class. Audio + score + clock reactive. Original work
// for the Arbit Media Machine. License: CC0.
// ARBIT_PARAM {"NAME":"sceneScale","TYPE":"float","DEFAULT":1.0,"MIN":0.5,"MAX":2.0}

float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float templeCell(vec3 p)
{
    // A finite stone block with three orthogonal tunnels removed. `max(box,-hole)`
    // is the signed box-minus-cross estimator; keeping the outer box is essential
    // for sphere tracing to see stone instead of an unbounded positive field.
    float carve = 0.32 + 0.10 * clamp(arbitBand(1.0) * 2.2 + arbitBand(3.0), 0.0, 1.0);
    float outer = sdBox(p, vec3(1.45, 1.65, 1.45));
    float tunnelX = max(abs(p.y) - carve, abs(p.z) - carve * 0.72);
    float tunnelY = max(abs(p.x) - carve * 0.72, abs(p.z) - carve);
    float tunnelZ = max(abs(p.x) - carve, abs(p.y) - carve * 0.72);
    return max(outer, -min(tunnelX, min(tunnelY, tunnelZ)));
}

float map(vec3 p)
{
    // Uniform scene scaling: default 1.0 is exactly the historical appearance.
    p /= sceneScale;
    // Slow rotation of the whole temple around Y, plus a beat-timed fold.
    float rot = uTime * 0.08 + uBeat * 0.03;
    float cs = cos(rot), sn = sin(rot);
    p.xz = mat2(cs, -sn, sn, cs) * p.xz;

    vec3 world = p;
    p.y -= 0.18 + 0.12 * beatPulse(0.45);
    float d = templeCell(p);
    float scale = 1.0;
    for (int i = 0; i < 3; ++i)
    {
        // Repeated symmetric cells add readable Menger-like buttresses while each
        // copy remains finite. min() is a union of conservative signed estimators.
        scale *= 2.35;
        vec3 q = abs(p) - vec3(1.12, 0.88, 1.12) / scale;
        d = min(d, templeCell(q * scale) / scale);
    }
    // A thin finite floor is a union, evaluated from the un-folded world position.
    float floorD = sdBox(world - vec3(0.0, -1.78, 0.0), vec3(3.4, 0.08, 3.4));
    return min(d, floorD) * sceneScale;
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.0015, 0.0);
    return normalize(vec3(
        map(p + e.xyy) - map(p - e.xyy),
        map(p + e.yxy) - map(p - e.yxy),
        map(p + e.yyx) - map(p - e.yyx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Block D owns the camera origin and ray; curated camera controls therefore
    // affect preview and export through the same contract.
    vec3 ro;
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 100; ++i)
    {
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.0012) { hit = t; break; }
        t += d;
        if (t > uCamFar) break;
    }

    vec3 col = mix(vec3(0.008, 0.010, 0.02), vec3(0.04, 0.05, 0.09),
                   smoothstep(-0.4, 0.6, rd.y));

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);
        vec3 lig = normalize(vec3(0.35, 0.9, 0.2));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        float amb = 0.16 + 0.14 * n.y;
        col = vec3(0.26, 0.30, 0.40) * (amb + 0.65 * dif);

        // Transient light-pulse flooding down from the shaft top.
        if (uOnsetAge < 0.9)
            col += vec3(0.9, 0.85, 1.0) * exp(-p.y * 0.35)
                 * (1.0 - uOnsetAge / 0.9) * 1.4;

        // Depth fog toward the temple's stone color.
        float fog = 1.0 - exp(-hit * 0.075);
        col = mix(col, vec3(0.02, 0.024, 0.05), fog);
    }

    // Score runes: glowing quads orbiting mid-height, hue by pitch class.
    float pulse = 0.7 + 0.7 * uRMS + 1.1 * uOnset;
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float midi = arbitNotePitch(i);
        float ang = (midi / 12.0) * 6.2831853 / 12.0 + uBeat * 0.22;
        vec3 c = vec3(cos(ang) * 2.3, 1.15, sin(ang) * 2.3);
        // screen-space glow: project the rune center onto the ray
        vec3 w = c - ro;
        float proj = dot(w, rd);
        if (proj > 0.0)
        {
            float sd = length(w - rd * proj);
            float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
            vec3 hue = hsv2rgb(vec3(fract(0.66 - tt * 0.52), 0.9, 1.0));
            col += hue * exp(-sd * sd * 900.0) * arbitNoteVel(i) * pulse;
        }
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
