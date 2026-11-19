// Synaptic Bloom — a raymarched neural garden that grows from the score.
// Sounding notes sprout as glowing neurons; harmonic links fire synapses
// along their connections with beat-locked pulses traveling master to
// slave. The whole web drifts like plankton, brightens with the mix, and
// new connections literally reach out and connect on the beat. Audio +
// score + clock reactive. Original work for the Arbit Media Machine.
// License: CC0.

float hash12(vec2 p)
{
    vec3 q = fract(vec3(p.xyx) * 1031.0);
    q += dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

// Stable 3D position for a note row: seeded by midi + cents so the same
// pitch always lives in the same neighborhood.
vec3 notePos(int row)
{
    float midi = arbitNotePitch(row);
    float cents = arbitNoteCents(row);
    float h1 = hash12(vec2(midi * 3.1, cents * 0.7));
    float h2 = hash12(vec2(cents * 1.3 + 11.0, midi * 5.7));
    float ang = h1 * 6.2831853;
    float rad = 0.6 + h2 * 1.8;
    return vec3(cos(ang) * rad,
                (h2 - 0.5) * 2.4 + cents * 0.001,
                sin(ang) * rad)
         + vec3(sin(uTime * 0.21 + h1 * 9.0),
                cos(uTime * 0.17 + h2 * 7.0),
                sin(uTime * 0.19 + h1 * 5.0)) * 0.12;   // plankton drift
}

float sdSphere(vec3 p, float r){ return length(p) - r; }

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Camera floats through the web at its center height.
    vec3 ro = vec3(sin(uTime * 0.09) * 3.1,
                   cos(uTime * 0.07) * 0.7,
                   cos(uTime * 0.09) * 3.1);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);

    float t = 0.0, hit = -1.0;
    int hitRow = -1;
    for (int i = 0; i < 80; ++i)
    {
        vec3 p = ro + rd * t;
        float d = 1e9;
        int argRow = -1;
        int sounding = arbitSoundingCount();
        for (int k = 0; k < sounding; ++k)
        {
            int n = arbitSoundingRow(k);
            float grow = 0.06 + 0.16 * arbitNoteVel(n)
                       + 0.05 * beatPulse(0.5);
            float dn = sdSphere(p - notePos(n), grow);
            if (dn < d) { d = dn; argRow = n; }
        }
        if (d < 0.002) { hit = t; hitRow = argRow; break; }
        t += d * 0.95;
        if (t > uCamFar) break;
    }

    // Deep organic backdrop: teal-black gradient with drifting haze blobs.
    vec3 col = mix(vec3(0.008, 0.014, 0.02), vec3(0.03, 0.07, 0.08),
                   smoothstep(-0.5, 0.7, rd.y));

    // Axons first (behind everything): glow lines between linked notes.
    for (int e = 0; e < uLinkCount; ++e)
    {
        vec4 L = arbitLinkTexel(e);
        int slave = int(L.x), master = int(L.y);
        if (slave < 0 || slave >= uNoteCount || master < 0 || master >= uNoteCount) continue;
        if (arbitNoteActive(slave) < 0.5 && arbitNoteActive(master) < 0.5) continue;

        vec3 a = notePos(master), b = notePos(slave);
        // Beat-locked signal pulse traveling master -> slave each bar.
        float travel = fract(uBarPhase * max(1.0, floor(arbitLinkRatio(e) * 2.0)));
        vec3 pulseP = mix(a, b, travel);

        for (int s = 1; s < 12; ++s)
        {
            float f = float(s) / 12.0;
            vec3 bp = mix(a, b, f);
            vec3 w = bp - ro;
            float proj = dot(w, rd);
            if (proj <= 0.02 || proj > 22.0) continue;
            float sd = length(w - rd * proj);
            col += vec3(0.35, 0.75, 0.90) * exp(-sd * sd * 1400.0) * 0.35;
        }
        // The traveling pulse itself, brighter.
        vec3 wp = pulseP - ro;
        float projP = dot(wp, rd);
        if (projP > 0.02 && projP < 22.0)
        {
            float sdP = length(wp - rd * projP);
            col += vec3(0.85, 1.0, 0.95)
                 * exp(-sdP * sdP * 5200.0)
                 * (1.2 + 1.5 * beatPulse(0.3));
        }
    }

    // Neuron bodies.
    if (hit > 0.0 && hitRow >= 0)
    {
        vec3 p = ro + rd * hit;
        vec2 e = vec2(0.002, 0.0);
        vec3 c = notePos(hitRow);
        vec3 n = normalize(p - c);
        vec3 lig = normalize(vec3(0.3, 0.8, 0.45));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        float tt = clamp((arbitNotePitch(hitRow) - 24.0) / 72.0, 0.0, 1.0);
        bool isRoot = arbitNoteIsRoot(hitRow) > 0.5;
        vec3 hue = isRoot ? vec3(1.0, 0.82, 0.40)
                 : hsv2rgb(vec3(fract(0.42 - tt * 0.30), 0.7, 1.0));
        float spec = pow(clamp(dot(reflect(rd, n), lig), 0.0, 1.0), 36.0);
        col = hue * (0.30 + 0.55 * dif) + vec3(spec) * 0.7
            + hue * (0.25 + 1.0 * uRMS + 0.9 * uOnset);   // inner glow
        col *= exp(-hit * 0.05);
    }

    // Ambient dust motes so the space has depth even with one note.
    float dust = hash12(floor(fragCoord * 0.5) + floor(uTime * 8.0));
    col += vec3(0.10, 0.16, 0.18) * step(0.9985, dust) * (0.4 + uRMS);

    col = pow(clamp(col, 0.0, 1.0), vec3(0.80));
    fragColor = vec4(col, 1.0);
}
