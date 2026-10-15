R"GLSL(
uniform int uMotionMode;
uniform int uSeed;
uniform float uElapsed;
uniform float uDrag;
uniform float uAttraction;
uniform vec4 uAudioMotion; // RMS force, onset impulse, onset age, link spring strength
uniform vec3 uCollision; // mode, body radius, restitution
uniform int uGeometryCount;
uniform vec4 uGeometryAnchors[64];
uniform vec4 uBodyReplay[64];

float timelineHash(uint n)
{
    n = (n << 13U) ^ n;
    n = n * (n * n * 15731U + 789221U) + 1376312589U;
    return float(n & 0x7fffffffU) / float(0x7fffffff);
}

// Impulse and constant-force responses of x'' + drag*x' + attraction*x = force.
vec2 motionResponse(float t, float attraction)
{
    float h = uDrag * 0.5;
    if (attraction < 0.0001)
    {
        float impulse = uDrag < 0.0001 ? t : (1.0 - exp(-uDrag * t)) / uDrag;
        return vec2(impulse, uDrag < 0.0001 ? 0.5 * t * t : (t - impulse) / uDrag);
    }
    float q = h * h - attraction;
    if (abs(q) < 0.0001)
        return vec2(t * exp(-h * t), (1.0 - exp(-h * t) * (1.0 + h * t)) / attraction);
    if (q < 0.0)
    {
        float w = sqrt(-q);
        float impulse = exp(-h * t) * sin(w * t) / w;
        return vec2(impulse, (1.0 - exp(-h * t) * cos(w * t) - h * impulse) / attraction);
    }
    float w = sqrt(q), a = exp((-h + w) * t), b = exp((-h - w) * t);
    return vec2((a - b) / (2.0 * w),
        (1.0 + ((-h - w) * a - (-h + w) * b) / (2.0 * w)) / attraction);
}

vec4 timelineParticle(uint i)
{
    if (uMotionMode == 2) return i < 64U ? uBodyReplay[i] : vec4(-10.0, -10.0, 0.0, 0.0);
    if (uElapsed < 0.0) return vec4(-10.0, -10.0, 0.0, 0.0);
    int matches = 0;
    for (int r = 0; r < min(uNoteCount, 128); ++r)
    {
        vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0);
        vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);
        if (int(t1.z + 0.5) == uSpawnTrack && t0.y > 0.001 && t0.z >= 0.0 && t0.w > 0.0)
            ++matches;
    }
    if (matches == 0 && uGeometryCount == 0) return vec4(-10.0, -10.0, 0.0, 0.0);
    uint seed = uint(uSeed);
    int pick = min(int(timelineHash(i * 747U + seed) * float(matches)), matches - 1);
    vec4 note = vec4(60.0, 1.0, 0.0, 1.0);
    vec4 links = vec4(0.0);
    for (int r = 0; r < min(uNoteCount, 128); ++r)
    {
        vec4 t0 = texelFetch(uNotes, ivec2(0, r), 0);
        vec4 t1 = texelFetch(uNotes, ivec2(1, r), 0);
        if (int(t1.z + 0.5) == uSpawnTrack && t0.y > 0.001 && t0.z >= 0.0 && t0.w > 0.0)
        { if (pick-- == 0) { note = t0; links = texelFetch(uNotes, ivec2(2, r), 0); break; } }
    }
    float duration = max(uLifetime, 0.1);
    float age = mod(uElapsed + timelineHash(i * 97U + seed) * duration, duration);
    float angle = (timelineHash(i * 31U + seed) - 0.5) * (uGeometryCount>0 ? 6.2831853 : 2.2);
    float speed = (0.25 + note.y * 0.75) * uForce * (1.0 + uAudioMotion.x);
    vec2 velocity = vec2(sin(angle) / max(uAspect, 0.001), cos(angle)) * speed;
    float stiffness = uAttraction + uAudioMotion.w * links.z;
    vec2 response = motionResponse(age, stiffness);
    vec2 anchor = vec2(clamp((note.x - 36.0) / 60.0, 0.0, 1.0) * 0.8 + 0.1, 0.12);
    vec2 target = anchor;
    if (uGeometryCount>0) {
        int point=min(int(timelineHash(i*193U+seed)*float(uGeometryCount)),uGeometryCount-1);
        anchor=uGeometryAnchors[point].xy; target=uGeometryAnchors[point].zw;
    }
    vec2 position = anchor
        + velocity * response.x
        + (uAttraction*(target-anchor)+uAudioMotion.w * links.xy - vec2(0.0, uGravity)) * response.y;
    if (uAudioMotion.z <= uElapsed && uAudioMotion.z < duration)
        position.y += uAudioMotion.y * motionResponse(uAudioMotion.z, stiffness).x;
    if (uCollision.x > 0.5)
    {
        // Replay a bounded body from its emission age. Fixed 1/120-second
        // steps never read the previous image or particle buffer.
        position = anchor;
        vec2 bodyVelocity = velocity;
        float impulseTime = age - uAudioMotion.z;
        if (impulseTime == 0.0 && uAudioMotion.z <= uElapsed)
            bodyVelocity.y += uAudioMotion.y;
        float elapsed = 0.0;
        for (int step = 0; step < 1200; ++step)
        {
            float dt = min(1.0 / 120.0, age - elapsed);
            if (dt <= 0.0) break;
            if (impulseTime > elapsed && impulseTime <= elapsed + dt
                && uAudioMotion.z <= uElapsed)
                bodyVelocity.y += uAudioMotion.y;
            vec2 acceleration = -stiffness * (position - anchor)
                + uAttraction*(target-anchor)+uAudioMotion.w * links.xy - vec2(0.0, uGravity);
            bodyVelocity = (bodyVelocity + acceleration * dt) / (1.0 + uDrag * dt);
            position += bodyVelocity * dt;
            for (int axis = 0; axis < 2; ++axis)
            {
                if (position[axis] < uCollision.y)
                { position[axis] = uCollision.y; bodyVelocity[axis] = abs(bodyVelocity[axis]) * uCollision.z; }
                if (position[axis] > 1.0 - uCollision.y)
                { position[axis] = 1.0 - uCollision.y; bodyVelocity[axis] = -abs(bodyVelocity[axis]) * uCollision.z; }
            }
            elapsed += dt;
        }
    }
    return vec4(position, 1.0 - age / duration, fract(note.x / 12.0));
}
)GLSL"
