#pragma once

#include "particle_parameters.h"
#include "particle_history.h"
#include "particle_geometry.h"
#include "particle_solid_state.h"

namespace videorender
{
// This is the simulation authority for coupled particles on every backend.
// Planar GPU passes copy the XY states; solid draws consume the same replay's
// immutable full poses through the retained native scene renderer.
inline std::array<std::array<float, 4>, 64> replayParticleBodies(
    const ParticleParams& input, double clipSeconds, float aspect,
    const canonicalblockc::CanonicalBlockCFrame* frame, std::string* diagnostic = nullptr,
    ParticleSolidState* solidState = nullptr)
{
    if (diagnostic) diagnostic->clear();
    if (solidState) *solidState={};
    if (input.preparedBodies && (!solidState || input.preparedSolids)) {
        if (solidState) *solidState=*input.preparedSolids;
        return *input.preparedBodies;
    }
    auto p = input;
    std::array<std::array<float, 4>, 64> result {};
    for (auto& value : result) value = { -10, -10, 0, 0 };
    if (p.historicalReplay && p.history && p.history->parameterAt && p.historyNodeId>0)
    {
        double resetTime=p.resetTime;
        std::string error;
        if (!p.history->parameterAt("clip"+std::to_string(p.historyClipId)+"/visual"
            +std::to_string(p.historyNodeId)+"/resetTime",p.historyProjectSeconds,resetTime,error)
            || !std::isfinite(resetTime))
        { if (diagnostic) *diagnostic=error.empty() ? "Automated reset time is nonfinite":error; return result; }
        p.resetTime=std::clamp(static_cast<float>(resetTime),0.0f,86400.0f);
    }
    const double elapsed = clipSeconds - p.resetTime;
    if (!std::isfinite(elapsed) || elapsed < 0) return result;
    const float duration = std::clamp(p.lifetime, 0.1f, 10.0f);
    // The shared rational clock represents 120 Hz steps and common 44.1/48 kHz
    // audio sample rates exactly, including their 96/192 kHz multiples.
    constexpr auto ticksPerSecond = kParticleReplayTicksPerSecond, stepTicks = kParticleReplayStepTicks;
    const auto window = particleReplayWindow(p, clipSeconds);
    const auto ageTicks = window.age;
    const double age = p.historicalReplay ? double(ageTicks) / ticksPerSecond
        : (p.resetMode == 0 ? std::fmod(elapsed, duration) : std::min(elapsed, double(duration)));
    const auto startTicks = window.start;
    using Body=ParticleRigidBody;
    const bool solid=p.simulationSpace==1;
    struct Link { int a, b; float rest, weight; };
    std::array<Body, 64> bodies {};
    std::array<Link, arbitblockc::kMaxLinks> links {};
    int count = 0, linkCount = 0;
    const int capacity = std::clamp(p.count, 1, 64);
    float mass = std::clamp(p.bodyMass, 0.01f, 100.0f);
    double emissionBudget=0;
    float previousReset=0;
    float previousResetTime=p.resetTime;
    double previousParameterTime=double(startTicks)/ticksPerSecond-(input.historyProjectSeconds-clipSeconds);
    const auto updateInputs = [&](const canonicalblockc::CanonicalBlockCFrame* sourceFrame, float rms, bool preserve)
    {
    frame = sourceFrame;
    const auto previous = bodies;
    const int previousCount = count;
    count = 0; linkCount = 0;
    std::array<int, arbitblockc::kMaxNotes> rows {};
    int rowCount = 0;
    if (frame != nullptr)
    {
        for (int row = 0; row < std::clamp(frame->noteRows(), 0, arbitblockc::kMaxNotes); ++row)
        {
            const auto at = static_cast<std::size_t>(row * 16);
            const auto& values = frame->noteTextureValues;
            if (std::isfinite(values[at]) && std::isfinite(values[at + 1])
                && values[at + 1] > 0.001f && values[at + 2] >= 0 && values[at + 3] > 0
                && values[at + 6] == static_cast<float>(p.spawnTrack)
                && frame->noteIdentities[static_cast<std::size_t>(row)] != 0)
                rows[static_cast<std::size_t>(rowCount++)] = row;
        }
        std::sort(rows.begin(), rows.begin() + rowCount, [frame](int a, int b)
        { return frame->noteIdentities[static_cast<std::size_t>(a)]
               < frame->noteIdentities[static_cast<std::size_t>(b)]; });
    }
    const auto hash = [](uint32_t n)
    {
        n = (n << 13U) ^ n;
        n = n * (n * n * 15731U + 789221U) + 1376312589U;
        return float(n & 0x7fffffffU) / float(0x7fffffff);
    };
    const int sources = p.geometryCount > 0 ? std::clamp(p.geometryCount, 0, 64) : rowCount;
    for (int index = 0; index < std::min(capacity, sources); ++index)
    {
        const int row = p.geometryCount > 0 ? -1 : rows[static_cast<std::size_t>(index)];
        const float pitch = row >= 0 ? frame->noteTextureValues[static_cast<std::size_t>(row * 16)] : 60;
        const float velocity = row >= 0 ? std::clamp(frame->noteTextureValues[static_cast<std::size_t>(row * 16 + 1)], 0.0f, 1.0f) : 1;
        const float angle = (hash(static_cast<uint32_t>(index) * 31U + static_cast<uint32_t>(p.seed)) - 0.5f) * 2.2f;
        const float speed = (0.25f + velocity * 0.75f) * p.force * (1 + p.rmsGain * rms);
        Body body {};
        body.x = std::clamp((pitch - 36) / 60, 0.0f, 1.0f) * 0.8f + 0.1f;
        body.y = 0.12f; body.tx = body.x; body.ty = body.y;
        body.z=body.tz=solid ? p.spawnZ:0;
        if (p.geometryCount > 0)
        {
            const auto& point = p.geometryAnchors[static_cast<std::size_t>(index)];
            body.x = point[0]; body.y = point[1]; body.tx = point[2]; body.ty = point[3];
            body.geometryIdentity=p.geometryIdentities[static_cast<std::size_t>(index)];
            if (solid) { body.z=p.geometryDepths[index][0]; body.tz=p.geometryDepths[index][1]; }
        }
        body.vx = std::clamp(std::sin(angle) * speed / (solid ? 1.0f:std::max(aspect, 0.001f)) + p.impulseX * velocity / mass, -32.0f, 32.0f);
        body.vy = std::clamp(std::cos(angle) * speed + p.impulseY * velocity / mass, -32.0f, 32.0f);
        if (solid) {
            body.vz=std::clamp(p.impulseZ*velocity/mass,-32.0f,32.0f);
            const float factor=0.00872664626f;
            const float sx=std::sin(p.orientationDegrees[0]*factor),cx=std::cos(p.orientationDegrees[0]*factor);
            const float sy=std::sin(p.orientationDegrees[1]*factor),cy=std::cos(p.orientationDegrees[1]*factor);
            const float sz=std::sin(p.orientationDegrees[2]*factor),cz=std::cos(p.orientationDegrees[2]*factor);
            body.orientation={sx*cy*cz-cx*sy*sz,cx*sy*cz+sx*cy*sz,cx*cy*sz-sx*sy*cz,cx*cy*cz+sx*sy*sz};
            body.angular=p.angularVelocity;
        }
        body.hue = pitch / 12 - std::floor(pitch / 12);
        body.noteRow = row;
        body.identity = row >= 0 ? frame->noteIdentities[static_cast<std::size_t>(row)] : index;
        bool existed=false;
        if (preserve)
            for (int old = 0; old < previousCount; ++old)
                if (previous[static_cast<std::size_t>(old)].identity == body.identity
                    && previous[static_cast<std::size_t>(old)].geometryIdentity == body.geometryIdentity)
                {
                    const auto& saved = previous[static_cast<std::size_t>(old)];
                    body.x = saved.x; body.y = saved.y; body.vx = saved.vx; body.vy = saved.vy;
                    body.z=saved.z; body.vz=saved.vz; body.orientation=saved.orientation; body.angular=saved.angular;
                    existed=true;
                    break;
                }
        if (!existed && p.emissionRate>=0)
        {
            if (emissionBudget<1) continue;
            emissionBudget-=1;
        }
        bodies[static_cast<std::size_t>(count++)] = body;
    }
    if (frame != nullptr && p.geometryCount == 0 && p.linkSpring > 0)
    {
        std::array<int, arbitblockc::kMaxLinks> order {};
        const int rowsInLinks = std::clamp(frame->linkRows(), 0, arbitblockc::kMaxLinks);
        for (int i = 0; i < rowsInLinks; ++i) order[static_cast<std::size_t>(i)] = i;
        std::sort(order.begin(), order.begin() + rowsInLinks, [frame](int a, int b)
        { return frame->linkIdentities[static_cast<std::size_t>(a)] < frame->linkIdentities[static_cast<std::size_t>(b)]; });
        for (int i = 0; i < rowsInLinks; ++i)
        {
            const int row = order[static_cast<std::size_t>(i)];
            if (frame->linkIdentities[static_cast<std::size_t>(row)] == 0) continue;
            const auto at = static_cast<std::size_t>(row * 4);
            const auto& values = frame->linkTextureValues;
            int a = -1, b = -1;
            for (int j = 0; j < count; ++j)
            {
                if (values[at] == float(bodies[static_cast<std::size_t>(j)].noteRow)) a = j;
                if (values[at + 1] == float(bodies[static_cast<std::size_t>(j)].noteRow)) b = j;
            }
            if (a < 0 || b < 0 || a == b || !std::isfinite(values[at + 2])
                || !std::isfinite(values[at + 3]) || values[at + 2] <= 0 || values[at + 3] <= 0) continue;
            const float ratio = std::clamp(std::max(values[at + 2] / values[at + 3], values[at + 3] / values[at + 2]), 1.0f, 8.0f);
            const auto& ba = bodies[static_cast<std::size_t>(a)];
            const auto& bb = bodies[static_cast<std::size_t>(b)];
            links[static_cast<std::size_t>(linkCount++)] = { a, b,
                std::max(0.001f, (solid ? std::hypot(std::hypot(bb.tx-ba.tx,bb.ty-ba.ty),bb.tz-ba.tz)
                    : std::hypot(bb.tx-ba.tx,bb.ty-ba.ty)) * p.linkRestScale),
                1 + p.linkRatioInfluence * (ratio - 1) };
        }
    }
    };
    canonicalblockc::FrameProducer historicalProducer;
    std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> historicalFrame;
    ParticleHistoryAudio historicalAudio;
    ParticleGeometryFrame geometryFrame;
    std::int64_t geometryTick = std::numeric_limits<std::int64_t>::min();
    const auto sampleGeometry = [&](std::int64_t tick)
    {
        if (tick == geometryTick) return true;
        std::string error;
        if (!sampleParticleGeometry(p,tick,geometryFrame,error))
        { if (diagnostic) *diagnostic=error; return false; }
        geometryTick=tick;
        return true;
    };
    const auto sampleHistory = [&](std::int64_t tick, bool preserve)
    {
        if (!p.history || !p.history->score) {
            if (diagnostic) *diagnostic="Coupled replay requires the owned project score history";
            return false;
        }
        const visualdeformation::RationalFrameTime time {tick, static_cast<std::uint32_t>(ticksPerSecond), 1};
        const double seconds = double(time.frame) * time.rateDenominator / time.rateNumerator;
        std::string parameterError;
        if (!sampleParticleParameters(input,p,seconds,parameterError))
        { if (diagnostic) *diagnostic=parameterError; return false; }
        mass=std::clamp(p.bodyMass,.01f,100.0f);
        const double localTime=seconds-(input.historyProjectSeconds-clipSeconds);
        const bool resetEdge=(p.reset>=.5f && previousReset<.5f)
            || (localTime>=p.resetTime && (p.resetTime!=previousResetTime || previousParameterTime<p.resetTime));
        previousReset=p.reset; previousResetTime=p.resetTime;
        previousParameterTime=localTime;
        if (resetEdge) { preserve=false; emissionBudget=0; }
        historicalAudio = p.history->audioAt(seconds);
        historicalFrame.reset();
        if (seconds >= 0 && !p.history->score->notes.empty())
        {
            auto key = p.history->key;
            key.frame = time.frame; key.fps = time.rateNumerator;
            const float beat = static_cast<float>(p.history->timeline.secondsToBeat(seconds));
            key.beat = beat;
            historicalFrame = historicalProducer.evaluate(key, p.history->score, beat);
            if (!historicalFrame) {
                if (diagnostic) *diagnostic="Coupled replay could not evaluate the canonical score at its fixed-step time";
                return false;
            }
        }
        if (!sampleGeometry(tick)) return false;
        updateInputs(historicalFrame.get(), historicalAudio.rms, preserve);
        return true;
    };
    if (p.historicalReplay)
    {
        if (!sampleHistory(startTicks, false)) return result;
    }
    else updateInputs(frame, p.rms, false);
    float radius = std::clamp(p.bodyRadius, 0.001f, 0.1f);
    const auto contact = [&](Body& a, Body& b)
    {
        const float dx = b.x - a.x, dy = b.y - a.y;
        float nx = 1, ny = 0, penetration = 0;
        if (p.bodyShape == 1)
        {
            const float px = 2 * radius - std::abs(dx), py = 2 * radius - std::abs(dy);
            if (px <= 0 || py <= 0) return;
            if (px <= py) { nx = dx < 0 ? -1.0f : 1.0f; penetration = px; }
            else { nx = 0; ny = dy < 0 ? -1.0f : 1.0f; penetration = py; }
        }
        else
        {
            const float distance = std::hypot(dx, dy);
            if (distance >= 2 * radius) return;
            if (distance > 1.0e-6f) { nx = dx / distance; ny = dy / distance; }
            penetration = 2 * radius - distance;
        }
        a.x -= nx * penetration * 0.5f; a.y -= ny * penetration * 0.5f;
        b.x += nx * penetration * 0.5f; b.y += ny * penetration * 0.5f;
        const float relative = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny;
        if (relative >= 0) return;
        const float impulse = -(1 + p.restitution) * relative * 0.5f;
        const float tangent = std::clamp(((b.vx - a.vx) * -ny + (b.vy - a.vy) * nx) * -0.5f,
                                         -p.friction * impulse, p.friction * impulse);
        a.vx -= nx * impulse - ny * tangent; a.vy -= ny * impulse + nx * tangent;
        b.vx += nx * impulse - ny * tangent; b.vy += ny * impulse + nx * tangent;
    };
    const auto walls = [&](Body& body)
    {
        if (p.collisionMode == 0) return;
        if (body.x < radius) { body.x = radius; body.vx = std::abs(body.vx) * p.restitution; body.vy *= 1 - p.friction; }
        if (body.x > 1 - radius) { body.x = 1 - radius; body.vx = -std::abs(body.vx) * p.restitution; body.vy *= 1 - p.friction; }
        if (body.y < radius) { body.y = radius; body.vy = std::abs(body.vy) * p.restitution; body.vx *= 1 - p.friction; }
        if (body.y > 1 - radius) { body.y = 1 - radius; body.vy = -std::abs(body.vy) * p.restitution; body.vx *= 1 - p.friction; }
    };
    const double impulseAt = age - p.onsetAge;
    const bool onsetInCycle = p.onsetAge <= elapsed && impulseAt >= 0;
    std::array<std::int64_t, 1201 + kParticleReplayMaxEvents> boundaries {};
    int boundaryCount = 0;
    if (p.historicalReplay)
    {
        boundaries[static_cast<std::size_t>(boundaryCount++)] = 0;
        for (std::int64_t tick = stepTicks; tick < ageTicks; tick += stepTicks)
            boundaries[static_cast<std::size_t>(boundaryCount++)] = tick;
        if (!visitParticleHistoryEvents(p, window, [&](std::int64_t tick)
            { boundaries[static_cast<std::size_t>(boundaryCount++)] = tick; })) {
            if (diagnostic) *diagnostic="Coupled replay exceeds 512 note/audio event boundaries";
            return result;
        }
        boundaries[static_cast<std::size_t>(boundaryCount++)] = ageTicks;
        std::sort(boundaries.begin(), boundaries.begin() + boundaryCount);
        boundaryCount = static_cast<int>(std::unique(boundaries.begin(), boundaries.begin() + boundaryCount) - boundaries.begin());
    }
    const int steps = p.historicalReplay ? boundaryCount - 1
        : std::min(1200, static_cast<int>(std::ceil(age * 120)));
    std::int64_t lastOnset = p.historicalReplay ? historicalAudio.onsetSample : -1;
    if (p.historicalReplay && lastOnset >= 0 && p.history->hasAudio()
        && std::llround(double(lastOnset) / p.history->audioSampleRate * ticksPerSecond) >= startTicks)
        lastOnset = -1; // An onset exactly on reset excites the new bodies once.
    for (int step = 0; step < steps; ++step)
    {
        const double start = p.historicalReplay ? double(boundaries[static_cast<std::size_t>(step)]) / ticksPerSecond : double(step) / 120;
        const float dt = p.historicalReplay
            ? static_cast<float>(double(boundaries[static_cast<std::size_t>(step + 1)] - boundaries[static_cast<std::size_t>(step)]) / ticksPerSecond)
            : static_cast<float>(std::min(1.0 / 120, age - start));
        bool historicalImpulse = false;
        if (p.historicalReplay)
        {
            if (!sampleHistory(startTicks + boundaries[static_cast<std::size_t>(step)], true)) return result;
            radius=std::clamp(p.bodyRadius,.001f,.1f);
            historicalImpulse = historicalAudio.onsetSample >= 0 && historicalAudio.onsetSample != lastOnset;
            lastOnset = historicalAudio.onsetSample;
        }
        emissionBudget=std::min(64.0,emissionBudget+std::max(0.0,double(p.emissionRate))*dt);
        const auto previousGeometry=geometryFrame;
        const auto previousBodies=bodies;
        std::array<std::array<float, 3>, 64> forces {};
        for (int index = 0; index < linkCount && p.linkConstraint == 0; ++index)
        {
            const auto& link = links[static_cast<std::size_t>(index)];
            const auto& a = bodies[static_cast<std::size_t>(link.a)];
            const auto& b = bodies[static_cast<std::size_t>(link.b)];
            const float dx = b.x - a.x, dy = b.y - a.y, dz=b.z-a.z;
            const float distance = solid ? std::hypot(std::hypot(dx,dy),dz):std::hypot(dx,dy);
            if (distance < 1.0e-6f) continue;
            const float force = std::clamp(p.linkSpring * link.weight * (distance - link.rest), -100.0f, 100.0f) / distance;
            forces[static_cast<std::size_t>(link.a)][0] += dx * force;
            forces[static_cast<std::size_t>(link.a)][1] += dy * force;
            forces[static_cast<std::size_t>(link.b)][0] -= dx * force;
            forces[static_cast<std::size_t>(link.b)][1] -= dy * force;
            forces[static_cast<std::size_t>(link.a)][2] += dz * force;
            forces[static_cast<std::size_t>(link.b)][2] -= dz * force;
        }
        for (int i = 0; i < count; ++i)
        {
            auto& body = bodies[static_cast<std::size_t>(i)];
            if (p.historicalReplay && historicalImpulse)
                body.vy += p.onsetGain * historicalAudio.onset / mass;
            if (!p.historicalReplay && onsetInCycle && ((step == 0 && impulseAt == 0) || (impulseAt > start && impulseAt <= start + dt)))
                body.vy += p.onsetGain * p.onset / mass;
            // Dense links and small masses share fixed numerical bounds, so a
            // hostile but admitted graph cannot grow an infinite trajectory.
            const float ax = std::clamp((forces[static_cast<std::size_t>(i)][0] + p.attraction * (body.tx - body.x)) / mass
                + (solid ? p.gravityX:0), -1000.0f, 1000.0f);
            const float ay = std::clamp((forces[static_cast<std::size_t>(i)][1] + p.attraction * (body.ty - body.y)
                + (p.historicalReplay ? p.rmsGain * historicalAudio.rms : 0)) / mass - p.gravity, -1000.0f, 1000.0f);
            body.vx = std::clamp((body.vx + ax * dt) / (1 + p.drag * dt), -32.0f, 32.0f);
            body.vy = std::clamp((body.vy + ay * dt) / (1 + p.drag * dt), -32.0f, 32.0f);
            body.x += body.vx * dt; body.y += body.vy * dt;
            if (solid) {
                const float az=std::clamp((forces[i][2]+p.attraction*(body.tz-body.z))/mass+p.gravityZ,-1000.0f,1000.0f);
                body.vz=std::clamp((body.vz+az*dt)/(1+p.drag*dt),-32.0f,32.0f);
                body.z+=body.vz*dt;
                solidRotate(body,dt,p.angularDrag);
            }
        }
        if (p.geometryBinding && !sampleGeometry(startTicks + boundaries[static_cast<std::size_t>(step + 1)])) return result;
        const auto geometryContact = [&](Body& body, const Body& previous, int index)
        {
            const auto& edge=geometryFrame.segments[static_cast<std::size_t>(index)];
            const auto& old=previousGeometry.segments[static_cast<std::size_t>(index)];
            const float ex=edge[2]-edge[0],ey=edge[3]-edge[1],length2=ex*ex+ey*ey;
            const float t=length2>1.0e-12f ? std::clamp(((body.x-edge[0])*ex+(body.y-edge[1])*ey)/length2,0.0f,1.0f) : 0;
            const float cx=edge[0]+t*ex,cy=edge[1]+t*ey;
            float dx=body.x-cx,dy=body.y-cy,distance=std::hypot(dx,dy);
            const float reach=radius+p.colliderThickness;
            const float motionX=((edge[0]-old[0])+(edge[2]-old[2]))*0.5f;
            const float motionY=((edge[1]-old[1])+(edge[3]-old[3]))*0.5f;
            const float fromX=previous.x+motionX,fromY=previous.y+motionY;
            const float travelX=body.x-fromX,travelY=body.y-fromY;
            const float cross=travelX*ey-travelY*ex;
            bool crossed=false, endpointHit=false;
            float firstHit=2;
            float hitX=cx,hitY=cy;
            float pointNormalX=1,pointNormalY=0;
            if (std::abs(cross)>1.0e-10f)
            {
                const float qx=edge[0]-fromX,qy=edge[1]-fromY;
                const float along=(qx*ey-qy*ex)/cross,segment=(qx*travelY-qy*travelX)/cross;
                crossed=along>=0 && along<=1 && segment>=0 && segment<=1;
                if (crossed) { firstHit=along; hitX=fromX+along*travelX; hitY=fromY+along*travelY; }
            }
            // Sweep the round end caps too. A fast body can touch a mesh
            // vertex without crossing the segment's center line.
            for (int endpoint=0;endpoint<(length2<=1.0e-12f ? 1 : 2);++endpoint)
            {
                const float px=edge[endpoint*2],py=edge[endpoint*2+1];
                const float qx=fromX-px,qy=fromY-py,a=travelX*travelX+travelY*travelY;
                const float b=2*(qx*travelX+qy*travelY),c=qx*qx+qy*qy-reach*reach;
                const float discriminant=b*b-4*a*c;
                if (a>1.0e-12f && discriminant>=0 && c>=0)
                {
                    const float along=(-b-std::sqrt(discriminant))/(2*a);
                    if (along>=0 && along<=1 && along<firstHit)
                    {
                        crossed=endpointHit=true; firstHit=along; hitX=px; hitY=py;
                        pointNormalX=(qx+along*travelX)/reach; pointNormalY=(qy+along*travelY)/reach;
                    }
                }
            }
            if (distance>=reach && !crossed) return;
            if (crossed || distance<1.0e-7f)
            {
                const float length=std::sqrt(length2);
                dx=length>1.0e-7f && !endpointHit ? -ey/length : pointNormalX;
                dy=length>1.0e-7f && !endpointHit ? ex/length : pointNormalY;
                if ((fromX-hitX)*dx+(fromY-hitY)*dy<0) { dx=-dx; dy=-dy; }
                body.x=hitX+dx*reach; body.y=hitY+dy*reach;
            }
            else
            { dx/=distance; dy/=distance; body.x=cx+dx*reach; body.y=cy+dy*reach; }
            const float surfaceX=std::clamp(motionX/dt,-32.0f,32.0f),surfaceY=std::clamp(motionY/dt,-32.0f,32.0f);
            const float relative=(body.vx-surfaceX)*dx+(body.vy-surfaceY)*dy;
            if (relative<0)
            {
                body.vx-=(1+p.restitution)*relative*dx; body.vy-=(1+p.restitution)*relative*dy;
                const float tangent=((body.vx-surfaceX)*-dy+(body.vy-surfaceY)*dx)*p.friction;
                body.vx+=tangent*dy; body.vy-=tangent*dx;
                body.vx=std::clamp(body.vx,-32.0f,32.0f); body.vy=std::clamp(body.vy,-32.0f,32.0f);
            }
        };
        for (int pass = 0; pass < 4; ++pass)
        {
            if (p.linkConstraint == 1)
                for (int i = 0; i < linkCount; ++i)
                {
                    const auto& link = links[static_cast<std::size_t>(i)];
                    auto& a = bodies[static_cast<std::size_t>(link.a)];
                    auto& b = bodies[static_cast<std::size_t>(link.b)];
                    const float dx = b.x - a.x, dy = b.y - a.y, dz=b.z-a.z;
                    const float distance = solid ? std::hypot(std::hypot(dx,dy),dz):std::hypot(dx,dy);
                    if (distance < 1.0e-6f) continue;
                    const float correction = (distance - link.rest) * 0.5f / distance;
                    a.x += dx * correction; a.y += dy * correction;
                    b.x -= dx * correction; b.y -= dy * correction;
                    a.z+=dz*correction; b.z-=dz*correction;
                    const float relative = ((b.vx - a.vx) * dx + (b.vy - a.vy) * dy+(b.vz-a.vz)*dz) * 0.5f / (distance * distance);
                    a.vx += dx * relative; a.vy += dy * relative;
                    b.vx -= dx * relative; b.vy -= dy * relative;
                    a.vz+=dz*relative; b.vz-=dz*relative;
                }
            if (p.collisionMode == 2)
            {
                for (int a = 0; a < count; ++a)
                    for (int b = a + 1; b < count; ++b)
                        if (solid) {
                            if (pass==0) solidSphereSweep(bodies[a],&bodies[b],solidPosition(previousBodies[a]),
                                solidPosition(previousBodies[b]),solidPosition(bodies[b]),p.bodyRadius,{},p);
                            solidSphereContact(bodies[a],&bodies[b],solidPosition(bodies[b]),p.bodyRadius,{},p);
                        } else contact(bodies[static_cast<std::size_t>(a)], bodies[static_cast<std::size_t>(b)]);
            }
            for (int i = 0; i < count; ++i)
            {
                if (solid) {
                    solidGeometryContact(bodies[i],previousBodies[i],geometryFrame,previousGeometry,dt,p,pass==0);
                    continue;
                }
                for (int edge=0;edge<geometryFrame.count;++edge)
                    geometryContact(bodies[static_cast<std::size_t>(i)],previousBodies[static_cast<std::size_t>(i)],edge);
                walls(bodies[static_cast<std::size_t>(i)]);
            }
        }
    }
    if (p.historicalReplay && !sampleHistory(startTicks + ageTicks, true)) return result;
    for (int i = 0; i < count; ++i)
    {
        const auto& body = bodies[static_cast<std::size_t>(i)];
        result[static_cast<std::size_t>(i)] = { body.x, body.y, 1, body.hue };
    }
    if (solidState) { solidState->bodies=bodies; solidState->count=count; solidState->colliders=geometryFrame; }
    return result;
}
}
