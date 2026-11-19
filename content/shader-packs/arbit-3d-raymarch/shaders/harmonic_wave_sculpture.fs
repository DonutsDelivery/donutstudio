// Harmonic Wave Sculpture — a raymarched kinetic sculpture of interlocking
// 3D sine ribbons in a gallery hall. Each sounding note contributes one
// ribbon whose twist frequency is its exact root-relative ratio, so a just
// major triad weaves locked helices into a braid while a wild tuning tangles
// them. The sculpture slowly rotates; transients kick a pulse of light down
// the braid; the pedestal mirror-polish doubles the whole piece. Works with
// no notes as a calm two-ribbon sculpture. Audio + score + clock reactive.
// Original work for the Arbit Media Machine. License: CC0.

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 1031.0);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

float sdCapsuleY(vec3 p, float h, float r)
{
    p.y -= clamp(p.y, -h, h);
    return length(p) - r;
}

// One helix strand around the Y axis. Distance approximated by sampling the
// helix curve at the height nearest the point.
float sdHelix(vec3 p, float turnsPerUnit, float phase, float radius, float tube)
{
    float y = clamp(p.y, -1.6, 1.6);
    // Iterate once to find a better closest point on the curve.
    for (int i = 0; i < 3; ++i)
    {
        float ang = y * turnsPerUnit * 6.2831853 + phase;
        vec3 c = vec3(cos(ang) * radius, y, sin(ang) * radius);
        y = clamp(p.y + (length(p.xz) - radius) * 0.0
                      + dot(p - c, vec3(0.0, 1.0, 0.0)), -1.6, 1.6);
    }
    float ang = y * turnsPerUnit * 6.2831853 + phase;
    vec3 c = vec3(cos(ang) * radius, y, sin(ang) * radius);
    return length(p - c) - tube;
}

float mapScene(vec3 p, out float strandId)
{
    float d = 1e9;

    // Score strands: one per active note, twist rate from exact ratio.
    int sounding = arbitSoundingCount();
    for (int k = 0; k < sounding; ++k)
    {
        int i = arbitSoundingRow(k);
        float cents = arbitNoteCents(i);
        float ratio = pow(2.0, cents / 1200.0);          // exact pitch ratio
        float turns = 0.9 * ratio;                        // twist per unit
        float phase = float(i) * 2.0944 + uTime * (0.25 + 0.35 * fract(ratio));
        float tubeR = 0.055 + 0.05 * arbitNoteVel(i);
        float di = sdHelix(p, turns, phase, 0.85, tubeR);
        if (di < d) { d = di; strandId = float(i); }
    }

    // Idle pair so the piece never vanishes.
    float ph1 = uTime * 0.3, ph2 = uTime * 0.3 + 3.14159;
    float idle = min(sdHelix(p, 1.5, ph1, 0.85, 0.06),
                     sdHelix(p, 1.5, ph2, 0.85, 0.06));
    if (idle < d) { d = idle; strandId = -1.0; }

    return d;
}

vec3 calcNormal(vec3 p, float strandRef)
{
    vec2 e = vec2(0.0018, 0.0);
    float sid;
    return normalize(vec3(
        mapScene(p + e.xyy, sid) - mapScene(p - e.xyy, sid),
        mapScene(p + e.yxy, sid) - mapScene(p - e.yxy, sid),
        mapScene(p + e.yyx, sid) - mapScene(p - e.yyx, sid)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Gallery viewpoint: slow orbit at sculpture height.
    float orbit = uTime * 0.14;
    vec3 ro = vec3(sin(orbit) * 2.9, 0.55 + 0.1 * sin(uTime * 0.19),
                   cos(orbit) * 2.9);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    float t = 0.0, hit = -1.0, sidHit = 0.0;
    for (int i = 0; i < 90; ++i)
    {
        vec3 p = ro + rd * t;
        float sid;
        float d = mapScene(p, sid);

        // Pedestal disc below the braid.
        float ped = max(length(p.xz) - 1.45, abs(p.y + 1.75) - 0.08);
        if (ped < d) { d = ped; sid = -2.0; }

        if (d < 0.0015) { hit = t; sidHit = sid; break; }
        t += d * 0.92;
        if (t > uCamFar) break;
    }

    // Gallery backdrop: dark walls, soft top light.
    vec3 col = mix(vec3(0.02, 0.02, 0.03), vec3(0.07, 0.07, 0.10),
                   smoothstep(-0.4, 0.7, rd.y));

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p, sidHit);
        vec3 lig = normalize(vec3(0.3, 0.9, 0.4));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        float spec = pow(clamp(dot(reflect(rd, n), lig), 0.0, 1.0), 48.0);

        if (sidHit > -0.5 && sidHit < 100.0)
        {
            int row = int(sidHit + 0.5);
            float tt = clamp((arbitNotePitch(row) - 24.0) / 72.0, 0.0, 1.0);
            vec3 hue = hsv2rgb(vec3(fract(0.60 - tt * 0.52), 0.85, 1.0));
            col = hue * (0.22 + 0.65 * dif) + vec3(1.0) * spec * 0.8;
            // Transient pulse of light racing down this strand.
            float pulsePos = fract(uOnsetAge * 1.4);
            float bandGlow = exp(-abs(p.y - (pulsePos * 3.2 - 1.6)) * 4.0)
                           * exp(-uOnsetAge * 2.0);
            col += hue * bandGlow * 2.5;
            // Emissive core so strands read as lit tubes, not painted metal.
            col += hue * (0.30 + 0.9 * uRMS);
        }
        else if (sidHit < -0.5 && sidHit > -1.5)
        {
            // Idle strands: silver steel.
            col = vec3(0.55, 0.58, 0.65) * (0.25 + 0.6 * dif) + vec3(spec) * 0.7;
            col += vec3(0.10, 0.12, 0.18) * (0.3 + 0.7 * uRMS);
        }
        else
        {
            // Pedestal: black granite with a faint reflection lift.
            float fres = pow(1.0 - max(dot(n, -rd), 0.0), 2.5);
            col = vec3(0.03) + vec3(0.06, 0.06, 0.09) * fres + vec3(spec) * 0.25;
        }
        col *= exp(-hit * 0.04);
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.80));
    fragColor = vec4(col, 1.0);
}
