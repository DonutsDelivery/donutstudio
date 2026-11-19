// Aurora Drift — flowing aurora curtains that scroll with the beat and bloom
// with the low end. Clock + audio reactive, no score needed. Original work for
// the Arbit Media Machine (uses Arbit dialect uniforms; not derived from any
// external shader). License: CC0.

float softMask(float inner, float outer, float value)
{
    return 1.0 - smoothstep(inner, outer, value);
}

// One aurora curtain: a horizontal band whose centre-height waves in x and time.
float curtain(vec2 uv, float baseY, float speed, float amp, float thick)
{
    float w = sin(uv.x * 6.2831853 * 1.5 + uBeat * speed) * 0.5
            + sin(uv.x * 6.2831853 * 0.5 - uBeat * speed * 0.6) * 0.5;
    float y = baseY + w * amp;
    return softMask(0.0, thick, abs(uv.y - y));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = fragCoord / uResolution.xy;

    float bass = arbitBand(4.0);
    float mids = arbitBand(20.0);
    float energy = clamp(uRMS * 4.0, 0.0, 1.0);

    // Night-sky vertical gradient, lifting slightly with overall energy.
    vec3 color = mix(vec3(0.010, 0.014, 0.030),
                     vec3(0.020, 0.010, 0.045), uv.y);
    color += vec3(0.02, 0.03, 0.06) * energy;

    // Three drifting curtains at different heights / hues.
    for (int i = 0; i < 3; ++i)
    {
        float fi = float(i);
        float baseY = 0.35 + fi * 0.18;
        float band = curtain(uv, baseY, 0.6 + fi * 0.25,
                             0.06 + 0.05 * bass, 0.10 + 0.04 * mids);
        float hue = fract(0.45 + fi * 0.12 + uBeat * 0.02);
        vec3 c = hsv2rgb(vec3(hue, 0.65, 0.9));
        // Vertical streak texture inside the curtain.
        float streak = 0.6 + 0.4 * sin(uv.x * 120.0 + uBeat * 2.0 + fi);
        color += c * band * streak * (0.35 + 0.65 * energy + 0.5 * bass);
    }

    // A faint star field that twinkles on transients.
    float stars = step(0.997, fract(sin(dot(floor(uv * 220.0),
                                           vec2(12.9898, 78.233))) * 43758.5453));
    color += vec3(0.8, 0.85, 1.0) * stars * (0.25 + 0.75 * uOnset);

    // Horizon glow rising from the bottom with the bass.
    color += vec3(0.10, 0.18, 0.16) * softMask(0.0, 0.45, uv.y) * (0.2 + bass);

    fragColor = vec4(pow(color, vec3(0.92)), 1.0);
}
