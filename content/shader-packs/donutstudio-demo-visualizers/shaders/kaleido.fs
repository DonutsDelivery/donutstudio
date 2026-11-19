/*{
  "DESCRIPTION": "Kaleido — a mirror-kaleidoscope of a drifting field. segments/zoom/spin and the tint are ISF INPUTS, so a kick can fan the segments and a bassline can zoom it once routed through the mod matrix. Original work for the Arbit Media Machine. License: CC0.",
  "CREDIT": "Arbit",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator", "Geometry"],
  "INPUTS": [
    { "NAME": "segments", "TYPE": "float", "DEFAULT": 6.0, "MIN": 2.0,  "MAX": 24.0 },
    { "NAME": "zoom",     "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.25, "MAX": 4.0 },
    { "NAME": "spin",     "TYPE": "float", "DEFAULT": 0.3, "MIN": -2.0, "MAX": 2.0 },
    { "NAME": "tint",     "TYPE": "color", "DEFAULT": [0.7, 0.9, 1.0, 1.0] }
  ]
}*/

#define PI 3.14159265

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * RENDERSIZE.xy) / RENDERSIZE.y;
    float r = length(uv);
    float a = atan(uv.y, uv.x);

    // Fold the angle into 'segments' mirrored wedges.
    float seg   = max(segments, 2.0);
    float wedge = PI / seg;
    a = abs(mod(a, 2.0 * wedge) - wedge);   // mirror within a wedge
    a += TIME * spin;                       // rotate the whole kaleido

    vec2 p = vec2(cos(a), sin(a)) * r * zoom;

    // A drifting source field, reflected through the wedges.
    float f = sin(p.x * 5.0 + TIME)
            + sin(p.y * 5.0 - TIME * 0.7)
            + sin((p.x + p.y) * 3.5 + TIME * 0.5);
    f = 0.5 + 0.5 * sin(f * 1.5);

    vec3 col = tint.rgb * f;
    col = mix(col, col.bgr, 0.5 + 0.5 * sin(r * 4.0 - TIME));   // colour cycling
    col *= 1.0 - smoothstep(0.6, 1.2, r);

    gl_FragColor = vec4(col, 1.0);
}
