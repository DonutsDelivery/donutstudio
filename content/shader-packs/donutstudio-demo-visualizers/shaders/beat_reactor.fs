// Beat Reactor. A precision-machined containment rig around a volatile plasma
// core. Bass loads the core, the spectrum travels around the field coils, and
// onsets discharge through the upper and lower pylons. Audio + clock reactive.
// Original work for DonutStudio. License: CC0.

const float PI = 3.14159265;
const float FAR_CLIP = 18.0;

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 0.1031);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float sdRoundBox(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float sdTorusY(vec3 p, vec2 t)
{
    return length(vec2(length(p.xz) - t.x, p.y)) - t.y;
}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r)
{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}

mat2 rot(float a)
{
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 rotateX(vec3 p, float a) { p.yz = rot(a) * p.yz; return p; }
vec3 rotateZ(vec3 p, float a) { p.xy = rot(a) * p.xy; return p; }

void reactorCameraRay(vec2 fragCoord, vec3 eye, out vec3 rd)
{
    vec2 uv = (fragCoord - 0.5 * uResolution) / uResolution.y;
    vec3 forward = normalize(uCamTarget - eye);
    vec3 right = normalize(cross(forward, uCamUp));
    vec3 up = cross(right, forward);
    float focal = 1.0 / tan(radians(uCamFov) * 0.5);
    rd = normalize(uv.x * right + uv.y * up + focal * forward);
}

float reactorMap(vec3 p, out float material)
{
    float d = 1e5;
    material = 0.0;

    // Three independently precessing containment rings form an atom-like
    // mechanism. Audio accelerates them and onsets kick their phase apart.
    float drive = clamp(uRMS * 1.8 + uPeak * 0.35, 0.0, 1.5);
    float kick = exp(-uOnsetAge * 4.0) * uOnset;
    vec3 q = rotateZ(rotateX(p, 0.42 + sin(uTime * 0.31) * 0.18),
                     uTime * (0.22 + drive * 0.18) + kick * 0.34);
    float coil = sdTorusY(q, vec2(1.48, 0.085));
    if (coil < d) { d = coil; material = 2.0; }

    q = rotateX(rotateZ(p, -0.72 + cos(uTime * 0.23) * 0.22),
                -uTime * (0.31 + drive * 0.13) - kick * 0.48);
    coil = sdTorusY(q, vec2(1.66, 0.075));
    if (coil < d) { d = coil; material = 2.0; }

    q = rotateZ(rotateX(p, 1.16 + sin(uTime * 0.19) * 0.20),
                uTime * (0.17 + drive * 0.22) + kick * 0.62);
    coil = sdTorusY(q, vec2(1.86, 0.065));
    if (coil < d) { d = coil; material = 2.0; }

    // Bright electron carriages run on different rings rather than pulsing the
    // whole assembly. Their positions expose the independent motion at a glance.
    for (int i = 0; i < 3; ++i)
    {
        float fi = float(i);
        float a = uTime * (0.72 + fi * 0.21 + drive * 0.34)
                + fi * 2.094 + kick * (0.8 + fi * 0.35);
        vec3 c = vec3(cos(a) * (1.48 + fi * 0.19), 0.0,
                      sin(a) * (1.48 + fi * 0.19));
        if (i == 0) c = rotateX(rotateZ(c, -uTime * 0.22), -0.42);
        if (i == 1) c = rotateZ(rotateX(c,  uTime * 0.31),  0.72);
        if (i == 2) c = rotateX(rotateZ(c, -uTime * 0.17), -1.16);
        float electron = length(p - c) - (0.12 + kick * 0.035);
        if (electron < d) { d = electron; material = 5.0; }
    }

    // Tapered-looking upper and lower power pylons, built from nested solids.
    float pylon = sdRoundBox(p - vec3(0.0, 2.05, 0.0), vec3(0.43, 0.72, 0.43), 0.09);
    pylon = min(pylon, sdRoundBox(p + vec3(0.0, 2.05, 0.0), vec3(0.43, 0.72, 0.43), 0.09));
    if (pylon < d) { d = pylon; material = 1.0; }

    float neck = sdCapsule(p, vec3(0.0, 1.30, 0.0), vec3(0.0, 1.68, 0.0), 0.23);
    neck = min(neck, sdCapsule(p, vec3(0.0, -1.30, 0.0), vec3(0.0, -1.68, 0.0), 0.23));
    if (neck < d) { d = neck; material = 2.0; }

    // A low circular service deck anchors the reactor in the chamber.
    float deck = max(length(p.xz) - 3.15, abs(p.y + 2.84) - 0.10);
    if (deck < d) { d = deck; material = 4.0; }

    return d;
}

vec3 reactorNormal(vec3 p)
{
    vec2 e = vec2(0.002, 0.0);
    float m;
    return normalize(vec3(
        reactorMap(p + e.xyy, m) - reactorMap(p - e.xyy, m),
        reactorMap(p + e.yxy, m) - reactorMap(p - e.yxy, m),
        reactorMap(p + e.yyx, m) - reactorMap(p - e.yyx, m)));
}

float softShadow(vec3 ro, vec3 rd, float maxT)
{
    float shade = 1.0;
    float t = 0.04;
    for (int i = 0; i < 24; ++i)
    {
        float m;
        float h = reactorMap(ro + rd * t, m);
        shade = min(shade, 13.0 * h / t);
        t += clamp(h, 0.025, 0.22);
        if (h < 0.001 || t > maxT) break;
    }
    return clamp(shade, 0.08, 1.0);
}

vec3 chamberBackdrop(vec3 ro, vec3 rd, float coreLight)
{
    // A cylindrical chamber remains coherent through the full camera orbit.
    float b = dot(ro.xz, rd.xz);
    float c = dot(ro.xz, ro.xz) - 8.6 * 8.6;
    float disc = b * b - dot(rd.xz, rd.xz) * c;
    if (disc <= 0.0) return vec3(0.003, 0.005, 0.009);
    float t = (-b + sqrt(disc)) / max(dot(rd.xz, rd.xz), 0.001);
    vec3 p = ro + rd * t;
    vec2 wallUV = vec2(atan(p.z, p.x) * 3.2, p.y * 0.62);
    vec2 cell = abs(fract(wallUV + vec2(0.2, 0.1)) - 0.5);
    float seam = smoothstep(0.455, 0.49, max(cell.x, cell.y));
    float recess = smoothstep(0.42, 0.12, max(cell.x, cell.y));
    vec3 col = mix(vec3(0.010, 0.014, 0.022), vec3(0.024, 0.030, 0.042), recess);
    col += vec3(0.05, 0.11, 0.16) * seam * (0.15 + 0.55 * beatPulse(0.30));
    col += vec3(1.0, 0.24, 0.06) * coreLight / (1.0 + 0.15 * dot(p.xy, p.xy));
    return col;
}

float plasmaDensity(vec3 p, float heat, out vec3 emission)
{
    float r = length(p);
    float bass = clamp(arbitBand(2.0) * 2.0 + arbitBand(5.0), 0.0, 2.0);
    float a = atan(p.z, p.x);
    float swirl = sin(a * 5.0 - uTime * (1.8 + bass) + p.y * 7.0);
    swirl += 0.55 * sin(a * 9.0 + uTime * 1.1 - r * 11.0);
    float shell = exp(-abs(r - (0.62 + 0.07 * swirl)) * 8.5);
    float heart = exp(-r * r * 5.8);
    float density = (0.32 + heat * 0.34) * shell + (0.48 + heat * 0.30) * heart;

    vec3 amber = vec3(1.0, 0.10, 0.012);
    vec3 whiteHot = vec3(1.0, 0.88, 0.55);
    emission = mix(amber, whiteHot, clamp(heart * 1.3 + heat * 0.18, 0.0, 1.0));
    return density;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // The camera completes an orbit instead of hovering around one hero angle.
    float orbit = -0.34 + uTime * 0.115;
    float cameraLift = 0.55 + sin(uTime * 0.19) * 0.62;
    float cameraRadius = 7.75 + sin(uTime * 0.13) * 0.30;
    vec3 ro = vec3(sin(orbit) * cameraRadius, cameraLift,
                   cos(orbit) * cameraRadius);
    vec3 rd;
    reactorCameraRay(fragCoord, ro, rd);

    float heat = clamp(uRMS * 3.2 + uPeak * 0.75, 0.0, 1.5);
    float coreFacing = pow(max(dot(rd, normalize(-ro)), 0.0), 30.0);
    vec3 col = chamberBackdrop(ro, rd, coreFacing * (0.15 + heat));

    // March the manufactured structure first. Its sharp specular response is
    // intentionally separate from the softer plasma integration below.
    float t = 0.0;
    float hitT = -1.0;
    float hitMaterial = 0.0;
    for (int i = 0; i < 104; ++i)
    {
        vec3 p = ro + rd * t;
        float material;
        float d = reactorMap(p, material);
        if (d < 0.0015) { hitT = t; hitMaterial = material; break; }
        t += d * 0.82;
        if (t > FAR_CLIP) break;
    }

    if (hitT > 0.0)
    {
        vec3 p = ro + rd * hitT;
        vec3 n = reactorNormal(p);
        vec3 keyDir = normalize(vec3(-0.7, 0.9, 0.5));
        vec3 coreDir = normalize(-p);
        float key = max(dot(n, keyDir), 0.0) * softShadow(p + n * 0.006, keyDir, 5.0);
        float core = max(dot(n, coreDir), 0.0) / (0.55 + dot(p, p));
        float rim = pow(1.0 - max(dot(n, -rd), 0.0), 4.0);
        vec3 viewDir = -rd;
        vec3 halfKey = normalize(keyDir + viewDir);
        float ndh = max(dot(n, halfKey), 0.0);
        float specSharp = pow(ndh, 110.0);
        float specWide = pow(ndh, 18.0);
        float facing = max(dot(n, viewDir), 0.0);
        float fresnel = pow(1.0 - facing, 5.0);
        vec3 reflected = reflect(-viewDir, n);
        float envBand = pow(0.5 + 0.5 * sin(atan(reflected.z, reflected.x) * 7.0
                                           + reflected.y * 9.0), 10.0);

        vec3 base = vec3(0.055, 0.070, 0.082);
        float metal = 0.82;
        if (hitMaterial > 1.5 && hitMaterial < 2.5)
            base = vec3(0.10, 0.125, 0.14);
        else if (hitMaterial > 2.5 && hitMaterial < 3.5)
        {
            base = vec3(0.20, 0.23, 0.24);
            metal = 0.35;
        }
        else if (hitMaterial > 3.5)
            base = vec3(0.018, 0.022, 0.028);

        if (hitMaterial > 4.5)
        {
            base = vec3(0.10, 0.42, 0.62);
            metal = 0.95;
        }

        col = base * (0.08 + key * 0.88 + core * 0.38);
        col += vec3(1.0, 0.17, 0.025) * core * (2.2 + heat * 2.0);
        col += mix(vec3(0.28, 0.42, 0.52), vec3(1.0, 0.91, 0.72), metal)
             * (specSharp * 1.7 + specWide * 0.22);
        col += vec3(0.05, 0.40, 0.66) * (fresnel * 0.62 + envBand * fresnel * 0.55);
        col += vec3(0.10, 0.28, 0.38) * rim * 0.42;

        if (hitMaterial > 4.5)
            col += vec3(0.12, 0.78, 1.0) * (1.8 + heat + exp(-uOnsetAge * 5.0));

        // Thin energized stripe along each field coil. It follows the spectrum
        // instead of making the entire model flash.
        if (hitMaterial > 1.5 && hitMaterial < 2.5)
        {
            float angle = fract(atan(p.z, p.x) / (2.0 * PI) + 0.5);
            float band = arbitBand(angle * 63.0);
            float stripe = pow(0.5 + 0.5 * sin(angle * 64.0 * PI), 12.0);
            col += mix(vec3(0.0, 0.35, 0.62), vec3(0.20, 0.95, 1.0), band)
                 * stripe * (0.16 + 1.8 * band);
        }
    }

    // Integrate only up to the first solid hit, so the plasma sits inside the
    // rig rather than bleeding through its metalwork.
    float maxVolT = hitT > 0.0 ? hitT : min(FAR_CLIP, 10.0);
    float jitter = hash12(fragCoord + floor(uTime * 30.0)) * 0.075;
    float coreB = dot(rd, ro);
    float coreC = dot(ro, ro) - 1.02 * 1.02;
    float coreDisc = coreB * coreB - coreC;
    float vt = coreDisc > 0.0 ? max(-coreB - sqrt(coreDisc), 0.0) + jitter : maxVolT;
    float trans = 1.0;
    vec3 volume = vec3(0.0);
    for (int i = 0; i < 80; ++i)
    {
        vec3 p = ro + rd * vt;
        if (length(p) < 1.02)
        {
            vec3 emission;
            float den = plasmaDensity(p, heat, emission);
            float alpha = clamp(den * 0.075, 0.0, 0.22);
            volume += trans * emission * den * 0.19;
            trans *= 1.0 - alpha;
            if (trans < 0.04) break;
        }
        vt += 0.075;
        if (vt > maxVolT) break;
    }
    col += volume * (1.15 + uOnset * 0.8);

    // The glass field is visible only at grazing angles. A pair of thin arcs
    // reads as containment without wrapping the scene in a translucent bubble.
    float b = dot(rd, ro);
    float c = dot(ro, ro) - 1.88 * 1.88;
    float disc = b * b - c;
    if (disc > 0.0)
    {
        float fieldT = -b - sqrt(disc);
        if (fieldT > 0.0 && (hitT < 0.0 || fieldT < hitT))
        {
            vec3 fp = ro + rd * fieldT;
            vec3 fn = normalize(fp);
            float fres = pow(1.0 - abs(dot(fn, rd)), 5.0);
            float latitude = pow(0.5 + 0.5 * sin(fp.y * 18.0 - uBeat * 1.4), 18.0);
            col += vec3(0.04, 0.52, 0.88) * fres * (0.22 + latitude)
                 * (0.35 + beatPulse(0.28));
        }
    }

    // Onsets send one clean discharge down the machine's vertical axis.
    float boltLife = exp(-uOnsetAge * 5.5);
    if (boltLife > 0.01)
    {
        vec3 axisA = vec3(0.0, -2.7, 0.0);
        vec3 axisB = vec3(0.0, 2.7, 0.0);
        vec3 ba = axisB - axisA;
        vec3 w = axisA - ro;
        float rr = dot(rd, rd), bb = dot(ba, ba), rb = dot(rd, ba);
        float rw = dot(rd, w), bw = dot(ba, w);
        float denom = rr * bb - rb * rb;
        float rayT = (rb * bw - bb * rw) / max(denom, 0.001);
        float lineT = clamp((rr * bw - rb * rw) / max(denom, 0.001), 0.0, 1.0);
        float boltD = length((ro + rd * max(rayT, 0.0)) - (axisA + ba * lineT));
        col += vec3(0.30, 0.78, 1.0) * exp(-boltD * boltD * 950.0) * boltLife * 2.8;
    }

    // Filmic shoulder keeps the core white-hot without flattening everything.
    col = max(col, vec3(0.0));
    col = col / (vec3(1.0) + col);
    col = pow(col, vec3(0.4545));
    vec2 uv = fragCoord / max(uResolution, vec2(1.0));
    float vignette = pow(clamp(16.0 * uv.x * uv.y * (1.0 - uv.x) * (1.0 - uv.y), 0.0, 1.0), 0.12);
    col *= mix(0.62, 1.0, vignette);
    fragColor = vec4(col, 1.0);
}