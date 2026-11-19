// Spectrum Rings — a radial spectrum analyzer. Distance from the centre selects
// a frequency band (bass at the core, treble at the rim) and that band's energy
// lights a ring at its radius, so the whole spectrum reads as a pulsing target.
// Slowly rotates with the beat. Audio + clock reactive. Original work for the
// Arbit Media Machine. License: CC0.

float pixelMask(float distanceValue, float widthPixels)
{
    float minDimension = max(min(uResolution.x, uResolution.y), 1.0);
    float radius = widthPixels / minDimension;
    float aa = max(fwidth(distanceValue) * 1.15, 0.65 / minDimension);
    return 1.0 - smoothstep(radius - aa, radius + aa, distanceValue);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / uResolution.xy;
    vec2 p = uv - 0.5;
    p.x *= uResolution.x / uResolution.y;

    float r = clamp(length(p) * 2.0, 0.0, 1.0);  // 0 centre .. 1 rim
    float a = atan(p.y, p.x) + uBeat * 0.25;      // rotates with the beat

    // Radius selects the band; gain lifts the typically-small band energies.
    float amp = clamp(arbitBand(r * 63.0) * 3.5, 0.0, 1.0);

    // Each radius glows by its own band's energy, with an angular shimmer.
    float shimmer = 0.65 + 0.35 * sin(a * 10.0 + r * 8.0);
    float glow = amp * shimmer;

    // A solid pixel-sized trace stays crisp at 4K; its halo is a separate
    // controlled light field rather than a wide normalized blur.
    float ringDistance = abs(r - amp);
    float ringCore = pixelMask(ringDistance, 1.15);
    float ringHalo = exp(-ringDistance * min(uResolution.x, uResolution.y) / 7.0)
                   * (1.0 - ringCore);

    vec3 hue = hsv2rgb(vec3(fract(0.58 - r * 0.55 + uBeat * 0.02), 0.75, 1.0));
    vec3 color = hue * (glow * 1.15 + ringCore * 1.25 + ringHalo * 0.22);
    color += vec3(1.12) * ringCore;

    // Bright pulsing core from overall loudness + transients.
    float core = exp(-pow(r * 6.0, 2.0));
    color += vec3(1.0, 0.7, 0.4) * core * (0.4 + 2.0 * uRMS + 1.0 * uOnset);

    // Radial spokes for a little structure even when quiet.
    float spokes = 0.5 + 0.5 * cos(a * 16.0);
    color += hue * spokes * (1.0 - r) * 0.06;

    // Faint backdrop so it never reads as pure black.
    color += vec3(0.02, 0.025, 0.04) * (1.0 - r * 0.5);

    fragColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(0.92)), 1.0);
}
