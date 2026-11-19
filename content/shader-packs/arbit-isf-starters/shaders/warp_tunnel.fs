/*{
  "DESCRIPTION": "Warp Tunnel — an endless ribbed tunnel that flies forward and flares with the mix. speed/twist/ribs and two colours are ISF INPUTS, so they become Arbit clip params (automatable + mod-matrix targets). Original work for the Arbit Media Machine. License: CC0.",
  "CREDIT": "Arbit",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Geometry"],
  "INPUTS": [
    { "NAME": "speed",  "TYPE": "float", "DEFAULT": 0.6, "MIN": 0.0,  "MAX": 3.0 },
    { "NAME": "twist",  "TYPE": "float", "DEFAULT": 1.0, "MIN": -4.0, "MAX": 4.0 },
    { "NAME": "ribs",   "TYPE": "float", "DEFAULT": 12.0, "MIN": 2.0, "MAX": 48.0 },
    { "NAME": "colorA", "TYPE": "color", "DEFAULT": [0.10, 0.45, 0.95, 1.0] },
    { "NAME": "colorB", "TYPE": "color", "DEFAULT": [0.95, 0.25, 0.55, 1.0] }
  ]
}*/

// Standard ISF: TIME / RENDERSIZE provided by the host; INPUTS arrive as uniforms.
// Arbit also exposes its musical prelude (uRMS, beatPulse, ...) to ISF shaders.

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * RENDERSIZE.xy) / RENDERSIZE.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // 1/r gives the fly-into-depth tunnel feel; the mix nudges the throttle.
    float depth = 1.0 / max(r, 0.06);
    float z = depth + TIME * speed + uRMS * 1.5;

    float rib  = 0.5 + 0.5 * sin(z * ribs + a * twist);   // ribs along depth
    float band = 0.5 + 0.5 * sin(a * 6.0 - z * 2.0);      // spiral bands

    float shade = rib * band;
    shade *= smoothstep(0.0, 0.25, r);          // dark mouth at the centre
    shade *= 1.0 - smoothstep(0.7, 1.3, r);     // vignette at the rim

    vec3 col = mix(colorA.rgb, colorB.rgb, band) * shade;
    col += colorA.rgb * (0.15 / (r + 0.2));     // glow down the throat
    col *= 0.7 + 0.6 * beatPulse(0.25);         // flash on the beat

    gl_FragColor = vec4(col, 1.0);
}
