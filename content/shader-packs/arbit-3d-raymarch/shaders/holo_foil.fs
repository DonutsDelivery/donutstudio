// Holo Foil — a kinetic embossed trading card with bounded real-time diffraction.
// Six visible wavelengths and four diffraction orders are evaluated against three
// crossing grating directions. The card rotates continuously rather than resetting
// on bar boundaries; audio changes the foil energy and embossed relief.

float hfHash(vec2 p)
{
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

mat2 hfRot(float a)
{
    float c = cos(a), s = sin(a);
    return mat2(c, -s, s, c);
}

vec3 hfToCard(vec3 p, float ax, float ay)
{
    p.xz = hfRot(-ay) * p.xz;
    p.yz = hfRot(-ax) * p.yz;
    return p;
}

vec3 hfFromCardVector(vec3 v, float ax, float ay)
{
    v.yz = hfRot(ax) * v.yz;
    v.xz = hfRot(ay) * v.xz;
    return v;
}

float hfRoundedBox(vec3 p, vec3 b, float r)
{
    vec3 q = abs(p) - b + r;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0) - r;
}

float hfCardMap(vec3 world, float ax, float ay)
{
    vec3 p = hfToCard(world, ax, ay);
    float relief = 0.008 * sin(p.x * 36.0 + sin(p.y * 11.0 + uTime * 0.35));
    relief += 0.005 * sin(length(p.xy) * 48.0 - uTime * 0.7);
    relief *= 0.45 + 0.75 * clamp(uRMS, 0.0, 1.0);
    p.z -= relief;
    return hfRoundedBox(p, vec3(1.18, 1.62, 0.055), 0.13);
}

vec3 hfNormal(vec3 p, float ax, float ay)
{
    const float e = 0.0025;
    return normalize(vec3(
        hfCardMap(p + vec3(e, 0.0, 0.0), ax, ay) - hfCardMap(p - vec3(e, 0.0, 0.0), ax, ay),
        hfCardMap(p + vec3(0.0, e, 0.0), ax, ay) - hfCardMap(p - vec3(0.0, e, 0.0), ax, ay),
        hfCardMap(p + vec3(0.0, 0.0, e), ax, ay) - hfCardMap(p - vec3(0.0, 0.0, e), ax, ay)));
}

vec3 hfWavelengthRgb(float wavelength)
{
    vec3 c = vec3(0.0);
    if (wavelength < 440.0) c = vec3((440.0 - wavelength) / 60.0, 0.0, 1.0);
    else if (wavelength < 490.0) c = vec3(0.0, (wavelength - 440.0) / 50.0, 1.0);
    else if (wavelength < 510.0) c = vec3(0.0, 1.0, (510.0 - wavelength) / 20.0);
    else if (wavelength < 580.0) c = vec3((wavelength - 510.0) / 70.0, 1.0, 0.0);
    else if (wavelength < 645.0) c = vec3(1.0, (645.0 - wavelength) / 65.0, 0.0);
    else c = vec3(1.0, 0.0, 0.0);

    float edge = 1.0;
    if (wavelength < 420.0) edge = 0.3 + 0.7 * (wavelength - 380.0) / 40.0;
    else if (wavelength > 650.0) edge = 0.3 + 0.7 * (700.0 - wavelength) / 50.0;
    return max(c * edge, 0.0);
}

vec3 hfDiffraction(vec3 n, vec3 viewDir, vec3 lightDir, vec3 tangent,
                   vec3 bitangent, vec2 uv)
{
    float wavelengths[6] = float[6](420.0, 465.0, 505.0, 550.0, 610.0, 675.0);
    vec3 spectral = vec3(0.0);
    float audioEnergy = 0.55 + 0.65 * clamp(uRMS + 0.7 * uOnset, 0.0, 1.2);
    float spacingNm = 780.0 + 140.0 * sin(uv.y * 18.0 + uTime * 0.22);

    for (int g = 0; g < 3; ++g)
    {
        float gf = float(g);
        float angle = gf * 2.0943951 + 0.32 * sin(uv.x * 8.0 + uv.y * 7.0 + gf);
        vec3 grating = normalize(cos(angle) * tangent + sin(angle) * bitangent);
        float projected = abs(dot(viewDir, grating) - dot(lightDir, grating));
        float region = 0.28 + 0.72 * smoothstep(0.18, 0.82,
            0.5 + 0.5 * sin(uv.x * (9.0 + gf * 4.0) + uv.y * (13.0 - gf * 2.0)));

        for (int w = 0; w < 6; ++w)
        {
            float lambdaNm = wavelengths[w];
            float ratio = projected * spacingNm / lambdaNm;
            float energy = 0.0;
            for (int order = 1; order <= 4; ++order)
            {
                float miss = ratio - float(order);
                energy += exp(-72.0 * miss * miss) / float(order);
            }
            spectral += hfWavelengthRgb(lambdaNm) * energy * region;
        }
    }

    float fresnel = pow(1.0 - clamp(dot(n, viewDir), 0.0, 1.0), 3.0);
    return spectral * (0.10 + 0.90 * fresnel) * audioEnergy / 4.2;
}

vec3 hfBackground(vec3 rd, vec2 fragCoord)
{
    vec2 uv = fragCoord / uResolution;
    vec3 color = mix(vec3(0.002, 0.004, 0.012), vec3(0.016, 0.005, 0.028), uv.y);
    float sweep = 0.5 + 0.5 * sin(atan(rd.y, rd.x) * 5.0 - uTime * 0.18);
    color += hsv2rgb(vec3(fract(0.64 + 0.16 * sweep), 0.72, 0.10 * sweep));

    vec2 cell = floor((uv - 0.5) * vec2(19.0, 11.0));
    vec2 local = fract((uv - 0.5) * vec2(19.0, 11.0)) - 0.5;
    float star = smoothstep(0.065, 0.0, length(local)) * step(0.84, hfHash(cell));
    color += vec3(0.30, 0.45, 0.85) * star * (0.45 + 0.55 * sin(uTime + hfHash(cell) * 6.2831));
    return color;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec3 ro, rd;
    arbitCameraRay(fragCoord, ro, rd);

    float ax = 0.16 * sin(uTime * 0.29) + 0.05 * sin(uBeat * 0.17);
    float ay = uTime * 0.18 + 0.22 * sin(uTime * 0.11);
    float travel = 0.0;
    float distanceToCard = 0.0;
    bool hit = false;
    for (int stepIndex = 0; stepIndex < 72; ++stepIndex)
    {
        vec3 p = ro + rd * travel;
        distanceToCard = hfCardMap(p, ax, ay);
        if (distanceToCard < 0.0015)
        {
            hit = true;
            break;
        }
        travel += max(distanceToCard * 0.72, 0.002);
        if (travel > 18.0) break;
    }

    vec3 color = hfBackground(rd, fragCoord);
    if (hit)
    {
        vec3 p = ro + rd * travel;
        vec3 local = hfToCard(p, ax, ay);
        vec2 uv = local.xy / vec2(2.36, 3.24) + 0.5;
        vec3 n = hfNormal(p, ax, ay);
        vec3 v = normalize(ro - p);
        vec3 l = normalize(vec3(-0.45, 0.72, 0.55));
        vec3 t = normalize(hfFromCardVector(vec3(1.0, 0.0, 0.0), ax, ay));
        t = normalize(t - n * dot(t, n));
        vec3 b = normalize(cross(n, t));

        float frame = 1.0 - smoothstep(0.0, 0.055,
            min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y)));
        float medallion = smoothstep(0.43, 0.12, length((uv - vec2(0.5, 0.57)) * vec2(1.0, 1.22)));
        float etched = 0.5 + 0.5 * sin(uv.y * 82.0 + sin(uv.x * 31.0) * 2.2);
        vec3 base = mix(vec3(0.025, 0.018, 0.055), vec3(0.10, 0.025, 0.16), medallion);
        base += vec3(0.08, 0.045, 0.12) * etched * medallion;
        base = mix(base, vec3(0.32, 0.20, 0.08), frame);

        float diffuse = max(dot(n, l), 0.0);
        float specular = pow(max(dot(reflect(-l, n), v), 0.0), 72.0);
        vec3 foil = hfDiffraction(n, v, l, t, b, uv);
        float engraving = smoothstep(0.46, 0.42, abs(length((uv - 0.5) * vec2(1.0, 0.75)) - 0.31));
        color = base * (0.20 + 0.80 * diffuse);
        color += foil * (0.45 + 0.55 * medallion);
        color += vec3(0.9, 0.72, 0.4) * (specular + engraving * 0.18);
        color += vec3(0.2, 0.4, 1.0) * uOnset * 0.16 * (1.0 - frame);
    }

    color = color / (1.0 + color);
    color = pow(max(color, 0.0), vec3(0.4545));
    fragColor = vec4(color, 1.0);
}
