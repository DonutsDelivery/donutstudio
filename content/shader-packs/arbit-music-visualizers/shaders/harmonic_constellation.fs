/*{
  "ISFVSN": "2.0",
  "DESCRIPTION": "A score-driven constellation on DonutStudio's prime lattice. Exact harmonic links become luminous edges, roots become anchors, and upcoming notes fade into place before they sound.",
  "CREDIT": "DonutStudio",
  "CATEGORIES": ["Score", "Just Intonation", "Generator"],
  "INPUTS": [
    { "NAME": "latticeScale", "TYPE": "float", "DEFAULT": 0.16, "MIN": 0.06, "MAX": 0.30, "LABEL": "Lattice scale" },
    { "NAME": "nodeSize", "TYPE": "float", "DEFAULT": 0.018, "MIN": 0.005, "MAX": 0.06, "LABEL": "Note size" },
    { "NAME": "linkWidth", "TYPE": "float", "DEFAULT": 0.008, "MIN": 0.001, "MAX": 0.03, "LABEL": "Link width" },
    { "NAME": "lookaheadFade", "TYPE": "float", "DEFAULT": 0.45, "MIN": 0.0, "MAX": 1.0, "LABEL": "Upcoming notes" },
    { "NAME": "noteColor", "TYPE": "color", "DEFAULT": [0.18, 0.68, 1.0, 1.0], "LABEL": "Note color" },
    { "NAME": "linkColor", "TYPE": "color", "DEFAULT": [0.62, 0.30, 1.0, 1.0], "LABEL": "Link color" },
    { "NAME": "rootColor", "TYPE": "color", "DEFAULT": [1.0, 0.68, 0.14, 1.0], "LABEL": "Root color" },
    { "NAME": "backgroundColor", "TYPE": "color", "DEFAULT": [0.006, 0.008, 0.018, 1.0], "LABEL": "Background" }
  ]
}*/

float pointSegmentDistance(vec2 point, vec2 start, vec2 end)
{
    vec2 segment = end - start;
    float amount = clamp(dot(point - start, segment) / max(dot(segment, segment), 1e-6), 0.0, 1.0);
    return length(point - (start + segment * amount));
}

float pixelMask(float distanceValue, float halfWidth)
{
    float aa = max(fwidth(distanceValue) * 1.15,
                   0.65 / max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0));
    return 1.0 - smoothstep(halfWidth - aa, halfWidth + aa, distanceValue);
}

float exactMidi(ArbitNote note)
{
    return 69.0 + 12.0 * log2(max(note.t1.x, 0.001) / 440.0);
}

vec2 notePosition(ArbitNote note)
{
    vec2 prime = lattice2(note.t2.y, note.t2.z);
    float higherPrime = note.t2.w * 0.18 + note.t3.x * 0.12;
    float pitchDisplacement = clamp((exactMidi(note) - note.t0.x) / 12.0, -1.0, 1.0);
    return vec2(0.5)
         + vec2(prime.x + higherPrime, prime.y - higherPrime * 0.35) * latticeScale
         + vec2(0.0, pitchDisplacement * latticeScale * 0.75);
}

float noteVisibility(ArbitNote note)
{
    float age = note.t0.z;
    float remaining = note.t0.w;
    if (remaining <= -0.5 || age < -uScoreLookaheadBeats) return 0.0;
    float upcoming = age < 0.0
        ? mix(lookaheadFade * 0.18, lookaheadFade, 1.0 - clamp(-age / max(uScoreLookaheadBeats, 0.01), 0.0, 1.0))
        : 1.0;
    float ending = smoothstep(-0.45, 0.12, remaining);
    return upcoming * ending;
}

vec2 restNodePosition(int e3, int e5, float aspect)
{
    vec2 position = vec2(0.5) + lattice2(float(e3), float(e5)) * latticeScale;
    position.x = 0.5 + (position.x - 0.5) * aspect;
    return position;
}

void main()
{
    vec2 uv = isf_FragNormCoord;
    vec2 point = uv;
    point.x = 0.5 + (point.x - 0.5) * (RENDERSIZE.x / RENDERSIZE.y);
    float aspect = RENDERSIZE.x / RENDERSIZE.y;
    float minDimension = max(min(RENDERSIZE.x, RENDERSIZE.y), 1.0);
    vec3 color = backgroundColor.rgb;

    float beatBreath = 0.6 + 0.4 * beatPulse(0.2);
    for (int gridLine = -5; gridLine <= 5; ++gridLine)
    {
        vec2 origin = vec2(0.5) + lattice2(float(gridLine), 0.0) * latticeScale;
        origin.x = 0.5 + (origin.x - 0.5) * aspect;
        float gridDistance = abs(point.x - origin.x);
        color += vec3(0.03, 0.045, 0.08)
               * (pixelMask(gridDistance, 0.65 / minDimension)
                  + exp(-gridDistance * minDimension / 3.0) * 0.16) * beatBreath;
    }

    // Resting state: a faint nebula and a placeholder prime lattice keep the
    // module readable before any note or harmonic link exists. Real notes and
    // links draw on top and the placeholder steps back once they arrive.
    float rest = uNoteCount == 0 ? 1.0 : 0.4;
    vec2 centered = point - vec2(0.5);
    color += vec3(0.20, 0.24, 0.50)
           * (0.5 + 0.5 * exp(-dot(centered, centered) * 5.0))
           * rest * (0.85 + 0.15 * beatBreath);
    float restLinkHalfWidth = max(linkWidth * 0.5, 0.8 / minDimension);
    for (int e3 = -4; e3 <= 4; ++e3)
    {
        for (int e5 = -3; e5 <= 3; ++e5)
        {
            vec2 here = restNodePosition(e3, e5, aspect);
            float hereDistance = length(point - here);
            float hereCore = pixelMask(hereDistance, nodeSize * 0.5);
            float hereHalo = exp(-hereDistance * minDimension / 16.0);
            color += (vec3(0.9) * hereCore + noteColor.rgb * (hereCore * 0.5 + hereHalo * 0.30))
                   * rest * (0.7 + 0.3 * beatBreath);
            if (e3 < 4)
            {
                float across = pointSegmentDistance(point, here, restNodePosition(e3 + 1, e5, aspect));
                color += linkColor.rgb * (pixelMask(across, restLinkHalfWidth) * 0.55
                         + exp(-across * minDimension / 6.0) * 0.10) * rest;
            }
            if (e5 < 3)
            {
                float upward = pointSegmentDistance(point, here, restNodePosition(e3, e5 + 1, aspect));
                color += linkColor.rgb * (pixelMask(upward, restLinkHalfWidth) * 0.55
                         + exp(-upward * minDimension / 6.0) * 0.10) * rest;
            }
        }
    }

    for (int edge = 0; edge < uLinkCount; ++edge)
    {
        ArbitLink link = arbitLink(edge);
        ArbitNote slave = arbitNote(int(link.t0.x + 0.5));
        ArbitNote master = arbitNote(int(link.t0.y + 0.5));
        float visibility = min(noteVisibility(slave), noteVisibility(master));
        if (visibility <= 0.001) continue;
        vec2 start = notePosition(slave);
        vec2 end = notePosition(master);
        start.x = 0.5 + (start.x - 0.5) * aspect;
        end.x = 0.5 + (end.x - 0.5) * aspect;
        float distanceToLink = pointSegmentDistance(point, start, end);
        float ratioHue = fract(log2(max(link.t0.z / max(link.t0.w, 1.0), 0.001)) * 0.35);
        vec3 ratioColor = mix(linkColor.rgb, hsv2rgb(vec3(ratioHue, 0.62, 1.0)), 0.4);
        float linkHalfWidth = max(linkWidth * 0.5, 0.8 / minDimension);
        float linkCore = pixelMask(distanceToLink, linkHalfWidth);
        float linkHalo = exp(-max(distanceToLink - linkHalfWidth, 0.0)
                             * minDimension / 5.0) * (1.0 - linkCore);
        color += ratioColor * (linkCore + linkHalo * 0.20)
               * visibility * (0.45 + 0.55 * beatBreath);
    }

    for (int row = 0; row < uNoteCount; ++row)
    {
        ArbitNote note = arbitNote(row);
        float visibility = noteVisibility(note);
        if (visibility <= 0.001) continue;
        vec2 position = notePosition(note);
        position.x = 0.5 + (position.x - 0.5) * aspect;
        float distanceToNote = length(point - position);
        float velocity = clamp(note.t0.y, 0.0, 1.0);
        vec3 pitchColor = hsv2rgb(vec3(fract(exactMidi(note) / 24.0), 0.65, 1.0));
        vec3 nodeColor = mix(noteColor.rgb, pitchColor, 0.42);
        nodeColor = mix(nodeColor, rootColor.rgb, clamp(note.t1.w, 0.0, 1.0));
        float nodeRadius = max(nodeSize * (0.58 + velocity * 0.18), 1.5 / minDimension);
        float core = pixelMask(distanceToNote, nodeRadius);
        float rim = max(0.0, pixelMask(distanceToNote,
                        nodeRadius + 1.4 / minDimension) - core);
        float halo = exp(-max(distanceToNote - nodeRadius, 0.0)
                         * minDimension / 7.0) * (1.0 - core);
        color += (vec3(1.2) * core + nodeColor * (core * 0.65 + rim * 1.6 + halo * 0.24))
               * visibility * (0.65 + velocity + uOnset * 0.35);
    }

    color = color / (color + vec3(0.72));
    gl_FragColor = vec4(pow(max(color, 0.0), vec3(0.84)), 1.0);
}
