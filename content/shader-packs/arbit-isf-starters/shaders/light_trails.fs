/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "Infinite light trails. A PERSISTENT feedback buffer is zoomed and rotated a hair every frame while a beat-driven source orbits through it, smearing into endless tunnels. The classic 'modern' programmatic visual — and a worked example of ISF multipass: pass 0 reads its own previous frame (TARGET trail, PERSISTENT) to write the new one, pass 1 tonemaps it to screen. decay/spin/zoom + tint are INPUTS, so every knob is an Arbit clip param (automatable + a mod-matrix destination); it also reads the musical prelude (uBeat, beatPulse, uRMS) so the song drives the motion.",
  "CREDIT": "Arbit",
  "CATEGORIES": ["Feedback", "Generator"],
  "INPUTS": [
    { "NAME": "decay", "TYPE": "float", "DEFAULT": 0.985, "MIN": 0.80, "MAX": 0.999, "LABEL": "Trail decay" },
    { "NAME": "spin",  "TYPE": "float", "DEFAULT": 0.04,  "MIN": -0.20, "MAX": 0.20, "LABEL": "Feedback spin" },
    { "NAME": "zoom",  "TYPE": "float", "DEFAULT": 0.985, "MIN": 0.95, "MAX": 1.02,  "LABEL": "Feedback zoom" },
    { "NAME": "tint",  "TYPE": "color", "DEFAULT": [0.4, 0.7, 1.0, 1.0],             "LABEL": "Source tint" }
  ],
  "PASSES": [
    { "TARGET": "trail", "PERSISTENT": true },
    {}
  ]
}*/

void main() {
    vec2 uv = isf_FragNormCoord;
    float aspect = RENDERSIZE.x / RENDERSIZE.y;

    if (PASSINDEX == 0) {
        // Sample the PREVIOUS frame's trail, rotated + zoomed about centre so the
        // image perpetually flows inward — the feedback that builds the tunnels.
        vec2 c = uv - 0.5;
        c.x *= aspect;
        float a = spin + 0.05 * sin(uBeat * 0.25);
        mat2 R = mat2(cos(a), -sin(a), sin(a), cos(a));
        vec2 src = R * c * zoom;
        src.x /= aspect;
        src += 0.5;
        float running = float(uPlaying);
        float frameDecay = pow(clamp(decay, 0.80, 0.999),
                               max(TIMEDELTA, 0.0) * 60.0 * running);
        vec3 prev = texture(trail, src).rgb * frameDecay;

        // Two bright sources orbiting on the beat (counter-rotating), hue drifting
        // over the bars; the beat pulse and the mix RMS both feed their energy. A
        // tight core plus a soft halo keeps the trails rich, not threadbare.
        float bass = clamp(arbitBand(4.0) * 4.0, 0.0, 1.0);
        float treble = clamp(arbitBand(50.0) * 5.0, 0.0, 1.0);
        float t = uBeat * 0.5;
        float energy = 0.65 + 0.55 * beatPulse(0.6) + uRMS * 0.65 + uOnset * 0.8;
        vec3 col = hsv2rgb(vec3(fract(uBeat * 0.04), 0.7, 1.0)) * tint.rgb;

        vec2 p0 = 0.5 + (0.22 + bass * 0.14) * vec2(cos(t), sin(t * 1.3));
        vec2 p1 = 0.5 + (0.18 + treble * 0.12)
                       * vec2(cos(-t * 1.7 + 2.0), sin(-t * 1.1 + 2.0));
        float d0 = distance(uv, p0);
        float d1 = distance(uv, p1);
        float core = (1.0 - smoothstep(0.0, 0.06, d0))
                   + 0.7 * (1.0 - smoothstep(0.0, 0.05, d1));
        float halo = 0.18 * (exp(-d0 * 7.0) + exp(-d1 * 8.0));
        vec3 add = col * energy * (core + halo) * running;

        gl_FragColor = vec4(max(prev, prev + add), 1.0);
    } else {
        // Show the trail buffer through a soft tonemap so highlights bloom.
        vec3 c = texture(trail, uv).rgb;
        c = c / (c + vec3(0.7));
        gl_FragColor = vec4(pow(max(c, 0.0), vec3(0.85)), 1.0);
    }
}
