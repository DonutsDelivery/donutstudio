// Harmonic Bloom — each sounding note opens a petal of a radial flower; the
// chord blooms, the petal hue follows pitch, brightness follows velocity and
// transients, and the whole bloom rotates with the beat. Score + audio + clock
// reactive. Original work for the Arbit Media Machine. License: CC0.

#define PI 3.14159265

float noteGate(ArbitNote n)
{
    float inGate = smoothstep(-0.20, 0.10, n.t0.z);
    float outGate = smoothstep(-0.45, 0.20, n.t0.w);
    return inGate * outGate;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / uResolution.xy;
    vec2 p = uv - 0.5;
    p.x *= uResolution.x / uResolution.y;

    float r = length(p);
    float ang = atan(p.y, p.x) + uBeat * 0.15;   // slow rotation

    vec3 color = mix(vec3(0.018, 0.010, 0.022),
                     vec3(0.010, 0.014, 0.028), uv.y);

    // Count active notes for petal spacing.
    int activeCount = 0;
    for (int i = 0; i < uNoteCount; ++i)
    {
        if (noteGate(arbitNote(i)) > 0.01) activeCount++;
    }
    float petals = max(float(activeCount), 1.0);

    for (int i = 0; i < uNoteCount; ++i)
    {
        ArbitNote n = arbitNote(i);
        float g = noteGate(n);
        if (g <= 0.01) continue;

        // Exact frequency ratio to the project root determines petal angle.
        // JI commas therefore move the flower instead of only changing metadata.
        float ratioPhase = fract(log2(max(n.t1.x, 1.0) / max(uRootFreq, 1.0)));
        float petalAng = ratioPhase * 2.0 * PI;
        float da = ang - petalAng;
        da = atan(sin(da), cos(da));            // wrap to -pi..pi

        float vel = clamp(n.t0.y, 0.0, 1.0);
        float petalLen = 0.18 + 0.22 * vel;     // bloom radius
        // Petal shape: lobe narrowing with radius, sharper with more petals.
        float lobe = cos(da * petals * 0.5);
        lobe = pow(max(lobe, 0.0), 3.0);
        float petal = lobe * (1.0 - smoothstep(0.0, petalLen, r));

        vec3 c = hsv2rgb(vec3(ratioPhase, 0.7, 1.0));
        c = mix(c, vec3(1.0, 0.9, 0.5), n.t1.w);   // root → gold

        color += c * petal * g * (0.5 + 0.7 * vel + 0.6 * uOnset);
    }

    // Glowing pistil at the centre, breathing with loudness.
    float core = exp(-pow(r * 9.0, 2.0));
    color += vec3(1.0, 0.85, 0.5) * core * (0.3 + 1.0 * uRMS + 0.5 * uOnset);

    fragColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(0.92)), 1.0);
}
