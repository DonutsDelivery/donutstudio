#pragma once

// The Realtime engine's fragment text and the scope that selects its program.
// Realtime rasterizes the same scene and materials with no ray walks through the
// glass BVH. A frame draws in phases: the opaque image, screen-space reflections
// over it, then each glass object over a copy of what is behind it. Every step
// count here is a quality constant of one approximation. None of them depends on
// the scene's size or content, and the marches scale with the camera's depth span.
// Preview and Export draw the same phases, so Accurate viewports match their export.
// A diffraction grating keeps the raster order sum between phases 0 and 1, so its
// highlights and spectral lines are real orders that reflections and glass can see.
// Only the receiver transport approximates, in backend_opengl.cpp, with a gather
// whose size is the square root of the scene's sample rows.

namespace arbitgpu::realtimeengine
{
inline bool& currentlyActive() noexcept
{
    thread_local bool value = false;
    return value;
}

// True while the renderer prepares and draws a Realtime frame on this thread.
inline bool active() noexcept { return currentlyActive(); }

class Scope final
{
public:
    Scope() noexcept : previous_ (currentlyActive()) { currentlyActive() = true; }
    ~Scope() { currentlyActive() = previous_; }
    Scope (const Scope&) = delete;
    Scope& operator= (const Scope&) = delete;

private:
    bool previous_;
};

// Follows the opaque visibility, sky, environment and pass composite text, and
// comes before the tail, which defines rotateQ and calls these from main.
inline const char* glslShader() noexcept
{
    return R"glsl(
uniform vec4 uRealtimeControl;
uniform sampler2D uRealtimeColor;
uniform sampler2D uRealtimeDepth;
uniform vec4 uCameraRotation;
uniform vec4 uProjection;
vec3 rotateQ(vec4 q, vec3 v);

// Lights have no size field, so each one is treated as this angular radius.
const float kRealtimeLightRadius = 0.03;
const float kRealtimeGoldenAngle = 2.39996323;

// Interleaved gradient noise. It is a function of the pixel only, so a repeated
// frame repeats exactly.
float realtimeNoise() {
    return fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
}

// Texture position in xy and linear depth in z, by the vertex shader's projection.
// A point at or behind the camera has a negative position.
vec3 realtimeProject(vec3 world) {
    vec3 camera = rotateQ(vec4(-uCameraRotation.xyz, uCameraRotation.w), world - uCameraTranslation);
    float distance = -camera.z;
    if (distance <= 0.0) return vec3(-1.0);
    vec2 ndc = vec2(camera.x / (uProjection.x * uProjection.y), camera.y / uProjection.x) / distance;
    return vec3(ndc * 0.5 + 0.5, (distance - uProjection.z) / max(uProjection.w - uProjection.z, 1e-6));
}

bool realtimeOnScreen(vec3 position) {
    return all(greaterThanEqual(position, vec3(0.0))) && all(lessThanEqual(position, vec3(1.0)));
}

bool realtimeSurfaceDepth(float depth) { return depth > 0.0 && depth < 1.0; }

// Marches the depth snapshot in steps that grow with distance. On a hit, the
// snapshot's color and a confidence that fades at the frame's edge come back.
bool realtimeTrace(vec3 origin, vec3 direction, out vec3 color, out float confidence) {
    color = vec3(0.0);
    confidence = 0.0;
    float span = max(uProjection.w - uProjection.z, 1e-6);
    float previous = 0.0;
    float travel = span * 0.001 * (0.5 + realtimeNoise());
    for (int step = 0; step < 40; ++step) {
        vec3 position = realtimeProject(origin + direction * travel);
        if (!realtimeOnScreen(position)) return false;
        float depth = texture(uRealtimeDepth, position.xy).r;
        float thickness = 2.0 * (travel - previous) / span;
        if (realtimeSurfaceDepth(depth) && position.z > depth + 0.0001 && position.z - depth <= thickness) {
            float low = previous;
            float high = travel;
            for (int refine = 0; refine < 5; ++refine) {
                float middle = 0.5 * (low + high);
                vec3 candidate = realtimeProject(origin + direction * middle);
                if (candidate.z > texture(uRealtimeDepth, candidate.xy).r) high = middle;
                else low = middle;
            }
            position = realtimeProject(origin + direction * high);
            color = texture(uRealtimeColor, position.xy).rgb;
            vec2 edge = min(position.xy, 1.0 - position.xy);
            confidence = clamp(min(edge.x, edge.y) * 12.0, 0.0, 1.0);
            return true;
        }
        previous = travel;
        travel *= 1.25;
    }
    return false;
}

vec3 realtimeFresnel(vec3 f0, vec3 normal, vec3 view) {
    return f0 + (vec3(1.0) - f0) * pow(1.0 - clamp(dot(normal, view), 0.0, 1.0), 5.0);
}

// Several rays across the light's disc, for every direct light kind.
float realtimeSoftVisibility(vec3 position, vec3 normal, vec3 toLight, float maxDistance) {
    vec3 axis = normalize(toLight);
    vec3 tangent = normalize(abs(axis.z) < 0.9 ? cross(axis, vec3(0.0, 0.0, 1.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(axis, tangent);
    float angle = realtimeNoise() * 6.28318531;
    float sum = 0.0;
    for (int ray = 0; ray < 4; ++ray) {
        float radius = kRealtimeLightRadius * sqrt((float(ray) + 0.5) / 4.0);
        angle += kRealtimeGoldenAngle;
        vec3 direction = normalize(axis + (tangent * cos(angle) + bitangent * sin(angle)) * radius);
        sum += opaqueVisibility(position, normal, direction, maxDistance, uOpaqueControl);
    }
    return sum * 0.25;
}

// The sky is a light from every direction, so its shadow is a hemisphere of rays.
float realtimeSkyVisibility(vec3 position, vec3 normal) {
    vec3 axis = normalize(normal);
    vec3 tangent = normalize(abs(axis.z) < 0.9 ? cross(axis, vec3(0.0, 0.0, 1.0)) : cross(axis, vec3(1.0, 0.0, 0.0)));
    vec3 bitangent = cross(axis, tangent);
    float angle = realtimeNoise() * 6.28318531;
    float sum = 0.0;
    for (int ray = 0; ray < 4; ++ray) {
        float radial = (float(ray) + 0.5) / 4.0;
        angle += kRealtimeGoldenAngle;
        vec3 direction = axis * sqrt(1.0 - radial) + (tangent * cos(angle) + bitangent * sin(angle)) * sqrt(radial);
        sum += opaqueVisibility(position, normal, normalize(direction), 3.402823e38, uOpaqueControl);
    }
    return sum * 0.25;
}

// Reflected scene color over the opaque image. The alpha takes out the environment
// reflection the base shading already holds where the march found an object.
vec4 realtimeReflection(vec3 position, vec3 normal, vec3 toView, vec3 base, float metallic, float roughness, float dielectric) {
    vec3 view = normalize(toView);
    vec3 direction = reflect(-view, normal);
    float gloss = (1.0 - roughness) * (1.0 - roughness);
    vec3 color;
    float confidence;
    if (gloss <= 0.0 || !realtimeTrace(position + normal * 0.0005 * (uProjection.w - uProjection.z), direction, color, confidence))
        return vec4(0.0);
    vec3 fresnel = realtimeFresnel(mix(vec3(dielectric), base, metallic), normal, view) * gloss * confidence;
    return vec4(fresnel * color, max(max(fresnel.r, fresnel.g), fresnel.b));
}

// The environment in a mirror direction, weighted only by the caller's Fresnel.
vec3 realtimeEnvironment(vec3 normal, vec3 toView, float roughness, float dielectric) {
    if (uSky[0].w <= 0.5) return vec3(0.0);
    return uSky[0].w > 1.5
        ? environmentImageBase(vec3(1.0), 1.0, roughness, dielectric, normal, toView, uSky[0], uSky[1])
        : analyticSkyBase(vec3(1.0), 1.0, roughness, dielectric, normal, toView, uSky[0], uSky[1], uSky[2], uSky[3], uSky[4]);
}

// One refracted direction per wavelength, each landing on the snapshot where the
// scene behind the glass lies. The depth gap sets how far the bent ray travels.
vec3 realtimeRefraction(vec3 position, vec3 normal, vec3 view, float roughness) {
    float span = max(uProjection.w - uProjection.z, 1e-6);
    vec3 own = realtimeProject(position);
    if (own.x < 0.0) return vec3(0.0);
    float ownDepth = texture(uRealtimeDepth, own.xy).r;
    bool surface = realtimeSurfaceDepth(ownDepth) && ownDepth > own.z;
    // Over the sky, nothing gives a gap, so the bend is judged at the camera's distance.
    float gap = surface ? (ownDepth - own.z) * span : own.z * span + uProjection.z;
    const vec3 wavelengths = vec3(0.6563, 0.5461, 0.4358);
    float blur = roughness * roughness * 0.05;
    float angle = realtimeNoise() * 6.28318531;
    vec3 result = vec3(0.0);
    for (int channel = 0; channel < 3; ++channel) {
        float wavelength = wavelengths[channel];
        float ior = uSurfaceCoating.y + uGlassTransport.y * (1.0 / (wavelength * wavelength) - 1.0 / (0.58756 * 0.58756));
        vec3 bent = uGlassTransport.x > 1.5 ? -view : refract(-view, normal, 1.0 / max(ior, 1e-3));
        if (dot(bent, bent) == 0.0) bent = reflect(-view, normal);
        vec3 target = realtimeProject(position + bent * gap);
        // A second read follows the ray to the depth where it lands.
        if (target.x >= 0.0) {
            float landed = texture(uRealtimeDepth, target.xy).r;
            if (realtimeSurfaceDepth(landed) && landed > own.z) {
                vec3 second = realtimeProject(position + bent * ((landed - own.z) * span));
                if (second.x >= 0.0) target = second;
            }
        } else target = own;
        float sum = 0.0;
        for (int tap = 0; tap < 4; ++tap) {
            angle += kRealtimeGoldenAngle;
            float radius = blur * sqrt((float(tap) + 0.5) / 4.0);
            sum += texture(uRealtimeColor, target.xy + vec2(cos(angle), sin(angle)) * radius)[channel];
        }
        result[channel] = sum * 0.25;
    }
    return result;
}

// What a glass surface shows: the scene behind it, its own shaded layer where it
// does not transmit, and a Fresnel reflection of nearby objects and the environment.
vec3 realtimeGlass(vec3 position, vec3 shadedNormal, vec3 toView, vec3 base, float roughness, float dielectric, vec3 shaded, vec3 surface) {
    vec3 view = normalize(toView);
    vec3 normal = dot(shadedNormal, view) < 0.0 ? -shadedNormal : shadedNormal;
    float transmitted = uSurfaceCoating.w > 0.5 ? uSurfaceCoating.x : 1.0;
    vec3 fresnel = realtimeFresnel(vec3(dielectric), normal, view);
    vec3 reflected = realtimeEnvironment(normal, toView, roughness, dielectric);
    vec3 hit;
    float confidence;
    if (realtimeTrace(position + normal * 0.0005 * (uProjection.w - uProjection.z), reflect(-view, normal), hit, confidence))
        reflected = mix(reflected, hit, confidence);
    vec3 behind = realtimeRefraction(position, normal, view, roughness) * base;
    return mix(shaded, (vec3(1.0) - fresnel) * behind + surface, transmitted) + fresnel * reflected;
}

// How much light reaches the air between the camera and this surface, as a factor
// on the Depth Fog color: shadowed air goes dim, so objects cast visible shafts.
vec3 realtimeFogLight(vec3 position) {
    bool fogPass = false;
    for (int i = 0; i < int(uPassProgramControl.x); ++i)
        if (int(uPassModes[i].x) == 1) fogPass = true;
    if (!fogPass) return vec3(1.0);
    vec3 lit = vec3(0.0);
    float total = 0.0;
    float noise = realtimeNoise();
    for (int lightIndex = 0; lightIndex < uLightCount; ++lightIndex) {
        int kind = fixtureLightKind(lightIndex);
        if (kind == 2) continue;
        vec4 light = fixtureLightColor(lightIndex);
        float weight = max(max(light.r, light.g), light.b) * light.a;
        total += weight * 8.0;
        for (int step = 0; step < 8; ++step) {
            vec3 air = mix(uCameraTranslation, position, (float(step) + noise) / 8.0);
            vec3 delta = fixtureLightPosition(lightIndex).xyz - air;
            vec3 toLight = kind == 0 ? rotateQ(fixtureLightRotation(lightIndex), vec3(0.0, 0.0, 1.0)) : normalize(delta);
            float cone = 1.0;
            if (kind == 3) {
                vec3 spot = rotateQ(fixtureLightRotation(lightIndex), vec3(0.0, 0.0, -1.0));
                cone = smoothstep(fixtureLightCone(lightIndex).y, fixtureLightCone(lightIndex).x, dot(spot, -toLight));
            }
            if (cone <= 0.0) continue;
            float visible = opaqueVisibility(air, toLight, toLight, kind == 0 ? 3.402823e38 : length(delta), uOpaqueControl);
            lit += light.rgb / max(max(max(light.r, light.g), light.b), 1e-6) * weight * cone * visible;
        }
    }
    return total > 0.0 ? vec3(0.35) + 0.65 * lit / total : vec3(1.0);
}
)glsl";
}
} // namespace arbitgpu::realtimeengine
