/*{
  "DESCRIPTION": "Plasma — a flowing spectrum field: bass expands its cells, mids bend hue, treble writes fine ripples, and onsets flash the intersections. scale, speed, hue and brightness are editable ISF INPUTS. Original work for DonutStudio. License: CC0.",
  "CREDIT": "Arbit",
  "ISFVSN": "2",
  "CATEGORIES": ["Generator"],
  "INPUTS": [
    { "NAME": "scale",      "TYPE": "float", "DEFAULT": 3.0, "MIN": 0.5, "MAX": 12.0 },
    { "NAME": "speed",      "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 4.0 },
    { "NAME": "hueShift",   "TYPE": "float", "DEFAULT": 0.0, "MIN": 0.0, "MAX": 1.0 },
    { "NAME": "brightness", "TYPE": "float", "DEFAULT": 1.0, "MIN": 0.0, "MAX": 2.0 }
  ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float bass = clamp(arbitBand(3.0) * 4.0, 0.0, 1.0);
    float mids = clamp(arbitBand(20.0) * 4.5, 0.0, 1.0);
    float treble = clamp(arbitBand(52.0) * 5.0, 0.0, 1.0);
    vec2 centered = uv - 0.5;
    centered *= 1.0 + bass * 0.34;
    vec2 p = (centered + 0.5) * scale * 2.6;
    float t = TIME * speed + uBeat * 0.08;

    // Four overlapping sine fields, one radial — the classic plasma recipe.
    float v = sin(p.x + t + bass * 1.8)
            + sin((p.y + t) * (0.86 + mids * 0.16))
            + sin((p.x + p.y + t) * 0.7 + mids * 2.2);
    vec2 c = p + vec2((0.42 + bass * 0.24) * sin(t * 0.3),
                      (0.42 + mids * 0.22) * cos(t * 0.4));
    v += sin(length(c) * 1.5 + t);
    v += sin((p.x - p.y) * (5.0 + treble * 7.0) - t * 1.6) * treble * 0.38;
    v *= 0.25;

    float hue = fract(0.5 + v * 0.28 + hueShift + mids * 0.16 - bass * 0.06);
    vec3 col = hsv2rgb(vec3(hue, 0.72 + treble * 0.20, 1.0));
    float cells = 0.5 + 0.5 * sin(v * 2.2);
    float intersections = pow(1.0 - abs(sin(v * 3.6 + treble * 2.0)), 12.0);
    col *= 0.34 + cells * 0.82;
    col += vec3(1.15) * intersections * (0.16 + treble * 0.64 + uOnset * 1.2);
    col *= brightness * (0.52 + bass * 0.70 + mids * 0.38 + treble * 0.30
                        + uRMS * 0.38 + uOnset * 0.52 + 0.28 * (0.5 + 0.5 * v));

    gl_FragColor = vec4(col, 1.0);
}
