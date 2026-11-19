// Score Monolith — a raymarched 2001-style monolith slab standing in an
// endless mirrored plane, except the monolith is the score. Its face is
// carved into a bar-phase clock of glowing notches, one edge per sounding
// note lights up in exact-pitch order, and harmonic links draw laser beams
// between their notes' edges on the slab. When a new note strikes, its edge
// flares and casts real light onto the floor. Audio + score + clock
// reactive. Original work for the Arbit Media Machine. License: CC0.

float sdBox(vec3 p, vec3 b)
{
    vec3 q = abs(p) - b;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

float mapScene(vec3 p)
{
    // The monolith: tall slab at the origin.
    float d = sdBox(p - vec3(0.0, 1.5, 0.0), vec3(1.05, 1.5, 0.16));
    // Ground plane.
    d = min(d, p.y);
    return d;
}

vec3 calcNormal(vec3 p)
{
    vec2 e = vec2(0.002, 0.0);
    return normalize(vec3(
        mapScene(p + e.xyy) - mapScene(p - e.xyy),
        mapScene(p + e.yxy) - mapScene(p - e.yxy),
        mapScene(p + e.yyx) - mapScene(p - e.yyx)));
}

// Emissive overlay evaluated on surfaces (and through space) after the march:
// returns added light for the fragment's world point p (or ray point).
vec3 monolithLight(vec3 p, vec3 ro, vec3 rd)
{
    vec3 add = vec3(0.0);

    // Bar-clock notches: a ring of marks down the slab's front face whose
    // brightness sweeps with the bar phase.
    if (abs(p.z - 0.16) < 0.03 && abs(p.x) < 1.06 && p.y > 0.02 && p.y < 3.0)
    {
        float row = floor((3.0 - p.y) * 8.0);          // 24 rows bottom-up
        float markPhase = fract(row / uBeatsPerBar);
        float lit = smoothstep(0.04, 0.0, abs(markPhase - uBarPhase));
        float sweep = smoothstep(0.35, 0.65, sin((uBarPhase - markPhase) * 6.2831853) * 0.5 + 0.5);
        add += vec3(0.9, 0.95, 1.0) * max(lit, 0.40 + sweep * 0.20) * step(abs(p.x), 0.98);
    }
    return add;
}

// Dusk sky with a bright horizon band. The slab and the mirror plane both
// pick it up, so the scene reads at rest and the audio only adds to it.
vec3 skyColor(vec3 d)
{
    float h = clamp(d.y, 0.0, 1.0);
    vec3 horizon = vec3(0.46, 0.55, 0.78);
    vec3 zenith = vec3(0.07, 0.10, 0.22);
    return mix(horizon, zenith, pow(h, 0.5)) * (1.0 + 0.5 * uRMS + 0.4 * uOnset);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    // Slow Kubrick drift around the slab; camera params still respected via
    // the shared helper basis.
    float orbit = uTime * 0.10 + uBeat * 0.02;
    vec3 ro = vec3(sin(orbit) * 3.4, 1.25 + 0.15 * sin(uTime * 0.23),
                   cos(orbit) * 3.4);
    vec3 rd;
    arbitCameraRay(fragCoord, ro, rd);
    // The camera helper puts the eye at uCamPos, which by default sits on
    // the mirror plane, so every ray would hit the floor at t = 0. Raise the
    // eye to the middle of the slab.
    ro.y += 1.45;

    float t = 0.0, hit = -1.0;
    for (int i = 0; i < 90; ++i)
    {
        vec3 p = ro + rd * t;
        float d = mapScene(p);
        if (d < 0.0015) { hit = t; break; }
        t += d;
        if (t > uCamFar) break;
    }

    vec3 col = skyColor(rd);

    if (hit > 0.0)
    {
        vec3 p = ro + rd * hit;
        vec3 n = calcNormal(p);

        // Key light from above-behind, cool fill from the horizon.
        vec3 lig = normalize(vec3(0.4, 0.85, -0.35));
        float dif = clamp(dot(n, lig), 0.0, 1.0);
        vec3 fillDir = normalize(vec3(-0.5, 0.3, 0.8));
        float fill = clamp(dot(n, fillDir), 0.0, 1.0);
        float amb = 0.30 + 0.12 * n.y;
        vec3 baseCol;
        if (p.y <= 0.001)
        {
            // Mirror plane: dark gloss with fake sky reflection.
            float fres = pow(1.0 - max(dot(n, -rd), 0.0), 2.5);
            baseCol = skyColor(reflect(rd, n)) * (0.30 + 0.70 * fres)
                    + vec3(0.04, 0.05, 0.08);
            col = baseCol;
        }
        else
        {
            float slabFres = pow(1.0 - max(dot(n, -rd), 0.0), 3.0);
            baseCol = vec3(0.16, 0.17, 0.20) * (amb + 0.60 * dif + 0.45 * fill);
            float spec = pow(clamp(dot(reflect(rd, n), lig), 0.0, 1.0), 60.0);
            col = baseCol + vec3(spec) * 0.4
                + skyColor(reflect(rd, n)) * (0.10 + 0.55 * slabFres);

            // Note edges: each active note owns one vertical edge stripe on
            // the front face, ordered by pitch, flaring on its attack age.
            if (abs(n.z) > 0.7)
            {
                int sounding = arbitSoundingCount();
                for (int k = 0; k < sounding; ++k)
                {
                    int i = arbitSoundingRow(k);
                    float midi = arbitNotePitch(i);
                    float tt = clamp((midi - 24.0) / 72.0, 0.0, 1.0);
                    float xEdge = mix(-0.92, 0.92, fract(midi / 36.0));
                    float stripe = exp(-abs(p.x - xEdge) * abs(p.x - xEdge) * 900.0)
                                 * smoothstep(0.05, 0.4, p.y)
                                 * smoothstep(3.0, 1.2, p.y);
                    float flare = exp(-max(arbitNoteAge(i), 0.0) * 3.0);
                    vec3 hue = hsv2rgb(vec3(fract(0.62 - tt * 0.5), 0.8, 1.0));
                    col += hue * stripe * arbitNoteVel(i) * (0.5 + 2.5 * flare);
                }
                col += monolithLight(p, ro, rd);

                // Resting edges: with no notes yet, five placeholder pitch
                // edges breathe on the face. Real notes replace them.
                if (sounding == 0)
                {
                    for (int k = 0; k < 5; ++k)
                    {
                        float xRest = mix(-0.8, 0.8, (float(k) + 0.5) / 5.0);
                        float restStripe = exp(-abs(p.x - xRest) * abs(p.x - xRest) * 900.0)
                                         * smoothstep(0.05, 0.4, p.y)
                                         * smoothstep(3.0, 1.2, p.y);
                        float breath = 0.55 + 0.45 * sin(uTime * 0.9 + float(k) * 1.7);
                        vec3 restHue = hsv2rgb(vec3(fract(0.62 - float(k) * 0.11), 0.8, 1.0));
                        col += restHue * restStripe * 0.75 * breath;
                    }
                }
            }

            // Laser beams between linked notes: project each link as a glow
            // line across the slab face in screen/ray space.
            for (int e = 0; e < uLinkCount; ++e)
            {
                vec4 L = arbitLinkTexel(e);
                int slave = int(L.x), master = int(L.y);
                if (slave < 0 || slave >= uNoteCount || master < 0 || master >= uNoteCount) continue;
                float mS = arbitNotePitch(slave), mM = arbitNotePitch(master);
                if (!(arbitNoteActive(slave) > 0.5 || arbitNoteActive(master) > 0.5)) continue;
                float xA = mix(-0.92, 0.92, fract(mS / 36.0));
                float xB = mix(-0.92, 0.92, fract(mM / 36.0));
                float yA = clamp(mix(0.4, 2.6, clamp((mS - 24.0) / 72.0, 0.0, 1.0)), 0.4, 2.6);
                float yB = clamp(mix(0.4, 2.6, clamp((mM - 24.0) / 72.0, 0.0, 1.0)), 0.4, 2.6);
                // Sample points along the beam against the current ray.
                for (int s = 1; s < 8; ++s)
                {
                    float f = float(s) / 8.0;
                    vec3 bp = vec3(mix(xA, xB, f), mix(yA, yB, f), 0.19);
                    vec3 w = bp - ro;
                    float proj = dot(w, rd);
                    if (proj <= 0.0 || abs(proj - hit) > 0.4) continue;
                    float sd = length(w - rd * proj);
                    col += vec3(1.0, 0.85, 0.55)
                         * exp(-sd * sd * 2600.0)
                         * (0.5 + 0.8 * beatPulse(0.45));
                }
            }
        }

        // Floor light cast by the whole lit face: soft pool under the slab.
        if (p.y <= 0.001)
        {
            float pool = exp(-length(p.xz - vec2(0.0, 0.0)) * 0.8);
            col += vec3(0.20, 0.22, 0.30) * pool * (0.55 + 0.45 * uRMS + 0.6 * uOnset);
            col = mix(col, col * 1.06, 0.5);   // gentle lift so glass reads
        }
        else
        {
            col *= exp(-hit * 0.035);
        }
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.78));
    fragColor = vec4(col, 1.0);
}
