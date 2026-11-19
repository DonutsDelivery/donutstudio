// Pulse Grid — a perspective grid that flashes on every beat, accents the
// downbeat of each bar, and runs a sweep line along the timeline. Pure clock
// reactivity (no audio/score needed), with an optional loudness lift. Original
// work for the Arbit Media Machine. License: CC0.

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

    float beatPulseV = beatPulse(0.45);                     // flash at beat start
    float barPulse   = 1.0 - smoothstep(0.0, 0.30, uBarPhase); // stronger on downbeat
    float lift = 0.3 + 0.7 * clamp(uRMS * 3.0, 0.0, 1.0);

    vec3 color = mix(vec3(0.020, 0.010, 0.030),
                     vec3(0.010, 0.020, 0.035), uv.y);

    // Horizon: grid recedes toward y = 0.55.
    float horizon = 0.55;
    float depth = (uv.y - horizon);
    vec3 gridCol = mix(vec3(0.2, 0.7, 0.9), vec3(0.9, 0.3, 0.6), barPulse);

    // Vertical lines fanning out from a vanishing point.
    if (uv.y < horizon)
    {
        float persp = 1.0 / max(horizon - uv.y, 0.02);
        float vx = fract((uv.x - 0.5) * persp * 0.5 + 0.5);
        float verticalDistance = min(vx, 1.0 - vx);
        float vlines = pixelMask(verticalDistance, 0.85);
        // Horizontal lines getting denser toward the horizon.
        float hy = fract((horizon - uv.y) * persp * 0.30);
        float horizontalDistance = min(hy, 1.0 - hy);
        float hlines = pixelMask(horizontalDistance, 0.85);
        float grid = max(vlines, hlines);
        float fade = smoothstep(0.0, horizon, uv.y);     // brighter near horizon
        color += gridCol * grid * fade * (0.25 + 0.75 * beatPulseV) * lift;
    }
    else
    {
        // Sky: pulsing sun/halo on the beat.
        vec2 c = (uv - vec2(0.5, horizon + 0.12));
        c.x *= uResolution.x / uResolution.y;
        float d = length(c);
        float halo = exp(-pow(d * 5.0, 2.0));
        color += gridCol * halo * (0.3 + 1.0 * beatPulseV + 0.6 * barPulse) * lift;
        float rings = pixelMask(abs(d - fract(uBeat * 0.5) * 0.4), 1.0);
        color += gridCol * rings * 0.4 * beatPulseV;
    }

    // Timeline sweep moving with the beat.
    float sweep = fract(uBeat * 0.25);
    float sweepDistance = abs(uv.x - sweep);
    color += vec3(1.0, 0.8, 0.4)
           * (pixelMask(sweepDistance, 1.0)
              + exp(-sweepDistance * min(uResolution.x, uResolution.y) / 4.0) * 0.18)
           * (0.2 + 0.5 * beatPulseV);

    fragColor = vec4(pow(clamp(color, 0.0, 1.0), vec3(0.92)), 1.0);
}
