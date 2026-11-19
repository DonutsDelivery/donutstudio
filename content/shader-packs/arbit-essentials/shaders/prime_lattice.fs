// Prime Lattice — plots every sounding note on the just-intonation lattice by
// its 3-limit / 5-limit prime exponents, with harmonic links drawn as glowing
// edges. The Arbit-specific score visualization: pure JI sits on a clean grid,
// temperament smears it. Score + clock reactive. Original work for the Arbit
// Media Machine (uses Block C score uniforms). License: CC0.

float pixelMask(float distanceValue, float radius)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(uResolution.x, uResolution.y), 1.0));
    return 1.0 - smoothstep(radius - aa, radius + aa, distanceValue);
}

float noteGate(ArbitNote n)
{
    float inGate = smoothstep(-0.20, 0.10, n.t0.z);   // fade in on onset
    float outGate = smoothstep(-0.45, 0.20, n.t0.w);  // fade out near end
    return inGate * outGate;
}

float exactMidi(ArbitNote note)
{
    return 69.0 + 12.0 * log2(max(note.t1.x, 0.001) / 440.0);
}

// Screen position from the 3- and 5-limit exponents (octave-equivalent), centred.
vec2 latticePos(ArbitNote n)
{
    vec2 l = lattice2(n.t2.y, n.t2.z);  // e3, e5
    float pitchDisplacement = clamp((exactMidi(n) - n.t0.x) / 12.0, -1.0, 1.0);
    return vec2(0.5, 0.5) + l * 0.16 + vec2(0.0, pitchDisplacement * 0.12);
}

// Distance from point a to segment a..b (in aspect-corrected space).
float segDist(vec2 p, vec2 a, vec2 b)
{
    vec2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-6), 0.0, 1.0);
    return length(pa - ba * h);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / uResolution.xy;
    vec2 p = uv;
    p.x = 0.5 + (p.x - 0.5) * (uResolution.x / uResolution.y);
    float aspect = uResolution.x / uResolution.y;
    float minDimension = max(min(uResolution.x, uResolution.y), 1.0);

    vec3 color = mix(vec3(0.015, 0.018, 0.026), vec3(0.022, 0.020, 0.034), uv.y);

    // Faint reference grid breathing on the beat.
    float breathe = 0.5 + 0.5 * sin(uBeat * 1.5708);
    for (int gx = -3; gx <= 3; ++gx)
    {
        vec2 gp = vec2(0.5, 0.5) + lattice2(float(gx), 0.0) * 0.16;
        gp.x = 0.5 + (gp.x - 0.5) * aspect;
        float gridDistance = abs(p.x - gp.x);
        color += vec3(0.04, 0.05, 0.07)
               * (pixelMask(gridDistance, 0.65 / minDimension)
                  + exp(-gridDistance * minDimension / 3.0) * 0.15)
               * (0.3 + 0.2 * breathe);
    }

    // Harmonic links first (drawn under the nodes).
    for (int i = 0; i < uLinkCount; ++i)
    {
        ArbitLink link = arbitLink(i);
        ArbitNote a = arbitNote(int(link.t0.x + 0.5));
        ArbitNote b = arbitNote(int(link.t0.y + 0.5));
        float g = noteGate(a) * noteGate(b);
        if (g <= 0.001) continue;

        vec2 pa = latticePos(a); pa.x = 0.5 + (pa.x - 0.5) * aspect;
        vec2 pb = latticePos(b); pb.x = 0.5 + (pb.x - 0.5) * aspect;
        float d = segDist(p, pa, pb);
        // Ratio numerator tints the edge (simple-ratio = warmer).
        float ratio = link.t0.z / max(link.t0.w, 1.0);
        vec3 edge = hsv2rgb(vec3(fract(0.55 + ratio * 0.05), 0.5, 0.9));
        float linkCore = pixelMask(d, 0.8 / minDimension);
        float linkHalo = exp(-d * minDimension / 5.0) * (1.0 - linkCore);
        color += edge * (linkCore + linkHalo * 0.20) * g * 0.6;
    }

    // Note nodes.
    for (int i = 0; i < uNoteCount; ++i)
    {
        ArbitNote n = arbitNote(i);
        float g = noteGate(n);
        if (g <= 0.001) continue;

        vec2 pos = latticePos(n);
        pos.x = 0.5 + (pos.x - 0.5) * aspect;
        float d = length(p - pos);
        float vel = clamp(n.t0.y, 0.0, 1.0);

        vec3 c = hsv2rgb(vec3(fract(0.08 + exactMidi(n) / 24.0), 0.65, 0.95));
        c = mix(c, vec3(1.0, 0.85, 0.35), n.t1.w);  // root → gold

        float radius = 0.010 + 0.010 * vel;
        float core = pixelMask(d, radius);
        float rim = max(0.0, pixelMask(d, radius + 1.35 / minDimension) - core);
        float halo = exp(-max(d - radius, 0.0) * minDimension / 7.0) * (1.0 - core);
        color += (vec3(1.15) * core + c * (core * 0.68 + rim * 1.5 + halo * 0.22))
               * g * (0.6 + 0.8 * vel + 0.5 * uOnset);
    }

    fragColor = vec4(pow(color, vec3(0.92)), 1.0);
}
