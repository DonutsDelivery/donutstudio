#include "gl_loader.h"
#include "particle_engine.h"
#include "particle_body_replay.h"
#include "particle_geometry_fixture.h"
#include "shader_generator.h"
#include <GLFW/glfw3.h>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

namespace
{
uint32_t hashTexture(arbitgl::GlFuncs& gl, unsigned texture, int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers(1, &fbo);
    gl.BindFramebuffer(GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);
    std::vector<uint8_t> pixels((size_t) width * height * 4);
    glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.DeleteFramebuffers(1, &fbo);
    uint32_t hash = 2166136261u;
    for (const auto value : pixels) { hash ^= value; hash *= 16777619u; }
    return hash;
}

std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> oneNote(float midi = 60.0f,
                                                                 int linkNumerator = 0)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    arbitmod::Note note;
    note.id = 1; note.midiNote = midi; note.velocity = 127.0f;
    note.freqHz = 261.625565f; note.startBeat = 0.0f; note.lengthBeats = 4.0f;
    score->notes.push_back (note);
    if (linkNumerator > 0)
    {
        note.id = -102; note.midiNote = 84; note.freqHz *= 4; note.trackId = 1;
        score->notes.push_back(note);
        score->links.push_back({-301, 1, -102, linkNumerator, 2, 0});
    }
    canonicalblockc::FrameKey key;
    key.projectGeneration = 1; key.sourceGeneration = 1; key.helperGeneration = 1;
    key.backendGeneration = 1; key.deviceGeneration = 1; key.scoreGeneration = 1;
    key.beatMapGeneration = 1; key.fpsGeneration = 1; key.loopGeneration = 1;
    key.seekGeneration = 1; key.frame = 0;
    key.beat = 0.0; key.fps = 30.0;
    canonicalblockc::FrameProducer producer;
    return producer.evaluate (key, std::move (score), 0.0f);
}

struct GlState
{
    GLint program {}, activeTexture {}, readFbo {}, drawFbo {}, drawBuffer {}, vao {}, ssbo {};
    GLint textures[3] {}, blendSrcRgb {}, blendDstRgb {}, blendSrcAlpha {}, blendDstAlpha {};
    GLint viewport[4] {};
    GLfloat clear[4] {};
    GLboolean blend {}, pointSize {};

    static GlState capture(arbitgl::GlFuncs& gl)
    {
        GlState s;
        glGetIntegerv(GL_CURRENT_PROGRAM, &s.program);
        glGetIntegerv(GL_ACTIVE_TEXTURE, &s.activeTexture);
        glGetIntegerv(GL_READ_FRAMEBUFFER_BINDING, &s.readFbo);
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING, &s.drawFbo);
        glGetIntegerv(GL_DRAW_BUFFER0, &s.drawBuffer);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING, &s.vao);
        gl.GetIntegeri_v(GL_SHADER_STORAGE_BUFFER_BINDING, 0, &s.ssbo);
        glGetIntegerv(GL_BLEND_SRC_RGB, &s.blendSrcRgb);
        glGetIntegerv(GL_BLEND_DST_RGB, &s.blendDstRgb);
        glGetIntegerv(GL_BLEND_SRC_ALPHA, &s.blendSrcAlpha);
        glGetIntegerv(GL_BLEND_DST_ALPHA, &s.blendDstAlpha);
        glGetIntegerv(GL_VIEWPORT, s.viewport);
        glGetFloatv(GL_COLOR_CLEAR_VALUE, s.clear);
        s.blend = glIsEnabled(GL_BLEND);
        s.pointSize = glIsEnabled(GL_PROGRAM_POINT_SIZE);
        for (int unit = 0; unit < 3; ++unit)
        {
            gl.ActiveTexture(GL_TEXTURE0 + unit);
            glGetIntegerv(GL_TEXTURE_BINDING_2D, &s.textures[unit]);
        }
        gl.ActiveTexture((GLenum) s.activeTexture);
        return s;
    }

    bool operator==(const GlState& other) const
    {
        return program == other.program && activeTexture == other.activeTexture
            && readFbo == other.readFbo && drawFbo == other.drawFbo
            && drawBuffer == other.drawBuffer && vao == other.vao && ssbo == other.ssbo
            && std::equal(std::begin(textures), std::end(textures), std::begin(other.textures))
            && blendSrcRgb == other.blendSrcRgb && blendDstRgb == other.blendDstRgb
            && blendSrcAlpha == other.blendSrcAlpha && blendDstAlpha == other.blendDstAlpha
            && std::equal(std::begin(viewport), std::end(viewport), std::begin(other.viewport))
            && std::equal(std::begin(clear), std::end(clear), std::begin(other.clear))
            && blend == other.blend && pointSize == other.pointSize;
    }
};
}

int main()
{
    constexpr int width = 160, height = 90;
    if (! glfwInit()) return 77;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    auto* window = glfwCreateWindow(width, height, "visual.particles GL", nullptr, nullptr);
    if (window == nullptr) { glfwTerminate(); return 1; }
    glfwMakeContextCurrent(window);
    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions(gl, missing)) return 1;
    if (! arbitgl::loadGl43Functions(gl)) return 1;

    videorender::ShaderClock clock;
    clock.frame = 60; clock.timeSec = 1.0; clock.timeDelta = 1.0 / 60.0; clock.playing = true;
    videorender::ParticleParams params;
    params.count = 256; params.size = 5.0f; params.force = 1.25f;
    params.seed = 77; params.lifetime = 2.5f;
    params.red = 0.1f; params.green = 0.8f; params.blue = 0.4f; params.alpha = 0.75f;
    const auto notes = oneNote();

    videorender::ParticleEngine first, second;
    GLuint sentinelTextures[3] {}, sentinelFbos[2] {}, sentinelVao {}, sentinelSsbo {};
    glGenTextures(3, sentinelTextures);
    gl.GenFramebuffers(2, sentinelFbos);
    gl.GenVertexArrays(1, &sentinelVao);
    gl.GenBuffers(1, &sentinelSsbo);
    for (int unit = 0; unit < 3; ++unit)
    {
        gl.ActiveTexture(GL_TEXTURE0 + unit);
        glBindTexture(GL_TEXTURE_2D, sentinelTextures[unit]);
    }
    gl.ActiveTexture(GL_TEXTURE2);
    gl.BindFramebuffer(GL_READ_FRAMEBUFFER, sentinelFbos[0]);
    gl.BindFramebuffer(GL_DRAW_FRAMEBUFFER, sentinelFbos[1]);
    glDrawBuffer(GL_NONE);
    gl.BindVertexArray(sentinelVao);
    gl.BindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, sentinelSsbo);
    glViewport(7, 9, 101, 47);
    glClearColor(0.125f, 0.25f, 0.5f, 0.75f);
    glEnable(GL_BLEND);
    glEnable(GL_PROGRAM_POINT_SIZE);
    gl.BlendFuncSeparate(GL_ONE, GL_ZERO, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    const auto stateBefore = GlState::capture(gl);
    const auto a = first.render(&gl, clock, width, height, params, notes.get());
    const bool successStateRestored = GlState::capture(gl) == stateBefore;

    auto failingGl = gl;
    failingGl.CreateShader = [](GLenum) -> GLuint { return 0; };
    videorender::ParticleEngine failing;
    const auto failedTexture = failing.render(&failingGl, clock, width, height, params, notes.get());
    const bool failureStateRestored = GlState::capture(gl) == stateBefore;

    const auto b = second.render(&gl, clock, width, height, params, notes.get());
    const auto hashA = a != 0 ? hashTexture(gl, a, width, height) : 0;
    const auto hashB = b != 0 ? hashTexture(gl, b, width, height) : 0;

    videorender::ParticleParams legacy;
    legacy.count = videorender::ParticleEngine::kMaxParticles;
    videorender::ParticleEngine legacyA, legacyB, empty;
    const auto legacyTexA = legacyA.render(&gl, clock, width, height, legacy, notes.get());
    const auto legacyTexB = legacyB.render(&gl, clock, width, height, legacy, notes.get());
    const auto emptyTex = empty.render(&gl, clock, width, height, legacy, nullptr);
    const auto legacyHashA = legacyTexA ? hashTexture(gl, legacyTexA, width, height) : 0;
    const auto legacyHashB = legacyTexB ? hashTexture(gl, legacyTexB, width, height) : 0;
    const auto emptyHash = emptyTex ? hashTexture(gl, emptyTex, width, height) : 0;
    // Use the same native pool/raster path with varying canonical note/audio
    // frames. Timeline motion must not depend on the frames visited beforehand.
    videorender::ParticleParams reactive = params;
    reactive.motionMode = 1;
    reactive.force = 0.25f;
    reactive.gravity = 0.1f;
    reactive.drag = 0.7f;
    reactive.attraction = 1.5f;
    reactive.rmsGain = 1.0f;
    reactive.onsetGain = 0.3f;
    reactive.rms = 0.25f;
    reactive.onset = 0.5f;
    reactive.onsetAge = 0.2f;
    videorender::ParticleEngine seeked, sequential;
    clock.frame = 36; clock.timeSec = 0.6; clock.playing = false;
    const auto renderHash = [&](videorender::ParticleEngine& engine,
                                const videorender::ParticleParams& p,
                                const canonicalblockc::CanonicalBlockCFrame* frame)
    {
        const auto texture = engine.render(&gl, clock, width, height, p, frame);
        return texture != 0 ? hashTexture(gl, texture, width, height) : 0;
    };
    const auto directReactive = renderHash(seeked, reactive, notes.get());
    bool reactiveValid = directReactive != 0 && directReactive != emptyHash;
    const auto differentNotes = oneNote(77.0f);
    for (int frame = 0; frame <= 36; ++frame)
    {
        clock.frame = frame; clock.timeSec = static_cast<double>(frame) / 60.0;
        clock.playing = true;
        renderHash(sequential, reactive, frame < 36 ? differentNotes.get() : notes.get());
    }
    reactiveValid = reactiveValid && renderHash(sequential, reactive, notes.get()) == directReactive;
    clock.playing = false;
    reactiveValid = reactiveValid && renderHash(sequential, reactive, notes.get()) == directReactive;
    clock.frame = 600000; clock.timeSec = 10000.0;
    reactiveValid = reactiveValid && renderHash(seeked, reactive, differentNotes.get()) != 0;
    clock.frame = 36; clock.timeSec = 0.6;
    reactiveValid = reactiveValid && renderHash(seeked, reactive, notes.get()) == directReactive;
    seeked.resetSimulation(&gl);
    reactiveValid = reactiveValid && renderHash(seeked, reactive, notes.get()) == directReactive;
    auto changed = reactive;
    changed.rms = changed.onset = 0.0f;
    reactiveValid = reactiveValid && renderHash(seeked, changed, notes.get()) != directReactive;
    changed = reactive; changed.seed += 1;
    reactiveValid = reactiveValid && renderHash(seeked, changed, notes.get()) != directReactive;
    changed = reactive; changed.spawnTrack = 9;
    reactiveValid = reactiveValid && renderHash(seeked, changed, notes.get()) == emptyHash;
    changed = reactive; changed.resetTime = 1.0f;
    reactiveValid = reactiveValid && renderHash(seeked, changed, notes.get()) == emptyHash;
    changed.resetTime = 0.2f;
    const auto resetRelative = renderHash(seeked, changed, notes.get());
    clock.timeSec = 0.4;
    changed.resetTime = 0.0f;
    reactiveValid = reactiveValid && renderHash(seeked, changed, notes.get()) == resetRelative;
    for (const auto coefficients : { std::pair<float, float>{0.0f, 0.0f},
                                    {2.0f, 1.0f}, {4.0f, 1.0f} })
    {
        changed.drag = coefficients.first; changed.attraction = coefficients.second;
        const auto value = renderHash(seeked, changed, notes.get());
        reactiveValid = reactiveValid && value != 0 && value != emptyHash
            && value == renderHash(sequential, changed, notes.get());
    }
    clock.frame = 36; clock.timeSec = 0.6; clock.playing = false;
    const auto linkedNotes = oneNote(60, 3);
    changed = reactive;
    changed.linkSpring = 2.5f;
    const auto springHash = renderHash(seeked, changed, linkedNotes.get());
    bool springsValid = springHash != 0 && springHash != emptyHash && springHash != directReactive;
    for (int frame = 0; frame <= 36; ++frame)
    {
        clock.frame = frame; clock.timeSec = static_cast<double>(frame) / 60;
        clock.playing = true;
        renderHash(sequential, changed, frame < 36 ? differentNotes.get() : linkedNotes.get());
    }
    springsValid = springsValid && renderHash(sequential, changed, linkedNotes.get()) == springHash;
    clock.frame = 600000; clock.timeSec = 10000;
    renderHash(seeked, changed, differentNotes.get());
    clock.frame = 36; clock.timeSec = 0.6; clock.playing = false;
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) == springHash;
    seeked.resetSimulation(&gl);
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) == springHash;
    const auto changedRatio = oneNote(60, 5);
    springsValid = springsValid && renderHash(seeked, changed, changedRatio.get()) != springHash;
    changed.linkRatioInfluence = 0;
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) != springHash;
    changed.linkSpring = 0;
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) == directReactive;
    changed.linkSpring = 2.5f;
    springsValid = springsValid && renderHash(seeked, changed, notes.get()) == directReactive;
    changed.resetTime = 1;
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) == emptyHash;
    changed.resetTime = 0.2f;
    const auto springReset = renderHash(seeked, changed, linkedNotes.get());
    changed.resetTime = 0; clock.timeSec = 0.4;
    springsValid = springsValid && renderHash(seeked, changed, linkedNotes.get()) == springReset;
    clock.timeSec = 2; clock.frame = 120;
    changed = reactive; changed.collisionMode = 1; changed.bodyRadius = 0.04f;
    changed.restitution = 0.9f; changed.gravity = 2; changed.force = 1;
    const auto bodiesHash = renderHash(seeked, changed, linkedNotes.get());
    bool bodiesValid = bodiesHash != 0 && bodiesHash != emptyHash;
    clock.timeSec = 10000; clock.frame = 600000;
    renderHash(sequential, changed, differentNotes.get());
    clock.timeSec = 2; clock.frame = 120;
    bodiesValid = bodiesValid && renderHash(sequential, changed, linkedNotes.get()) == bodiesHash;
    seeked.resetSimulation(&gl);
    bodiesValid = bodiesValid && renderHash(seeked, changed, linkedNotes.get()) == bodiesHash;
    changed.restitution = 0;
    bodiesValid = bodiesValid && renderHash(seeked, changed, linkedNotes.get()) != bodiesHash;
    changed.restitution = 0.9f; changed.collisionMode = 0;
    bodiesValid = bodiesValid && renderHash(seeked, changed, linkedNotes.get()) != bodiesHash;
    changed=reactive; changed.geometryCount=2; changed.force=0.04f; changed.attraction=4;
    changed.geometryAnchors[0]={0.25f,0.35f,0.25f,0.35f};
    changed.geometryAnchors[1]={0.7f,0.65f,0.7f,0.65f};
    const auto geometryHash=renderHash(seeked,changed,nullptr);
    bool geometryValid=geometryHash!=0 && geometryHash!=emptyHash;
    clock.timeSec=100; clock.frame=6000; renderHash(sequential,changed,nullptr);
    clock.timeSec=2; clock.frame=120;
    geometryValid=geometryValid && renderHash(sequential,changed,nullptr)==geometryHash;
    changed.geometryAnchors[0][2]=0.6f; changed.geometryAnchors[1][3]=0.2f;
    geometryValid=geometryValid && renderHash(seeked,changed,nullptr)!=geometryHash;
    changed = {};
    changed.motionMode = 2; changed.count = 2; changed.geometryCount = 2;
    changed.lifetime = 4; changed.force = 0; changed.bodyRadius = 0.07f;
    changed.collisionMode = 2; changed.size = 12; changed.impulseY = 0.1f;
    changed.geometryAnchors[0] = {0.48f,0.5f,0.48f,0.5f};
    changed.geometryAnchors[1] = {0.52f,0.54f,0.52f,0.54f};
    clock.timeSec = 0.5; clock.frame = 30;
    const auto coupledHash = renderHash(seeked, changed, nullptr);
    bool coupledValid = coupledHash != 0 && coupledHash != emptyHash;
    clock.timeSec = 10000; clock.frame = 600000;
    renderHash(sequential, changed, nullptr);
    clock.timeSec = 0.5; clock.frame = 30;
    coupledValid = coupledValid && renderHash(sequential, changed, nullptr) == coupledHash;
    seeked.resetSimulation(&gl);
    coupledValid = coupledValid && renderHash(seeked, changed, nullptr) == coupledHash;
    // The rasterized centers must match the shared CPU replay positions.
    const auto cpu = videorender::replayParticleBodies(changed, clock.timeSec, float(width)/height, nullptr);
    const auto texture = seeked.render(&gl, clock, width, height, changed, nullptr);
    coupledValid = coupledValid && texture != 0;
    if (texture != 0)
    {
        std::vector<uint8_t> pixels(width * height * 4);
        glBindTexture(GL_TEXTURE_2D, texture);
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
        for (int i = 0; i < 2; ++i)
        {
            const int x = std::clamp(int(cpu[i][0] * width), 0, width - 1);
            const int y = std::clamp(int(cpu[i][1] * height), 0, height - 1);
            coupledValid = coupledValid && pixels[static_cast<std::size_t>((y * width + x) * 4 + 3)] != 0;
        }
    }
    changed.bodyShape = 1;
    coupledValid = coupledValid && renderHash(seeked, changed, nullptr) != coupledHash;
    auto history = std::make_shared<videorender::ParticleHistorySource>();
    history->score = std::make_shared<const arbitmod::Score>();
    std::vector<arbitblockb::FeatureFrame> historyAudio(100);
    for (std::size_t i = 0; i < historyAudio.size(); ++i)
    {
        historyAudio[i].sampleIndex = static_cast<std::int64_t>(i + 1) * arbitblockb::kHop;
        historyAudio[i].rms = 0.3f;
        if (i >= 10) { historyAudio[i].onset = 1; historyAudio[i].onsetAge = float(i - 10) * arbitblockb::kHop / 48000; }
    }
    history->audioFrames = &historyAudio; history->audioSampleRate = 48000;
    changed.historicalReplay = true; changed.history = history;
    changed.rmsGain = 0.1f; changed.onsetGain = 0.2f;
    changed.historyProjectSeconds = clock.timeSec;
    const auto historyHash = renderHash(seeked, changed, nullptr);
    bool historyValid = historyHash != 0 && historyHash != emptyHash;
    clock.timeSec = 10000; clock.frame = 600000; changed.historyProjectSeconds = clock.timeSec;
    renderHash(sequential, changed, nullptr);
    clock.timeSec = 0.5; clock.frame = 30; changed.historyProjectSeconds = clock.timeSec;
    historyValid = historyValid && renderHash(sequential, changed, nullptr) == historyHash;
    seeked.resetSimulation(&gl);
    historyValid = historyValid && renderHash(seeked, changed, nullptr) == historyHash;
    changed.onsetGain = 0;
    historyValid = historyValid && renderHash(seeked, changed, nullptr) != historyHash;
    std::string geometryReplayError;
    auto animatedCollider=particleGeometryFixture(true,geometryReplayError);
    bool animatedColliderValid=animatedCollider && prepareParticleGeometryFixture(*animatedCollider,0.5,geometryReplayError);
    if (animatedColliderValid)
    {
        const auto prepared=animatedCollider->preparedBodies;
        const auto colliderHash=renderHash(seeked,*animatedCollider,nullptr);
        animatedColliderValid=colliderHash!=0 && colliderHash!=emptyHash;
        prepareParticleGeometryFixture(*animatedCollider,3.7,geometryReplayError);
        prepareParticleGeometryFixture(*animatedCollider,0.5,geometryReplayError);
        seeked.resetSimulation(&gl);
        animatedColliderValid=animatedColliderValid && geometryReplayError.empty()
            && *animatedCollider->preparedBodies==*prepared
            && renderHash(seeked,*animatedCollider,nullptr)==colliderHash;
    }
    seeked.shutdown(&gl); sequential.shutdown(&gl);
    legacyA.shutdown(&gl); legacyB.shutdown(&gl); empty.shutdown(&gl);
    const auto firstLog = first.log();
    const auto secondLog = second.log();
    first.shutdown(&gl); second.shutdown(&gl); failing.shutdown(&gl);
    gl.DeleteBuffers(1, &sentinelSsbo);
    gl.DeleteVertexArrays(1, &sentinelVao);
    gl.DeleteFramebuffers(2, sentinelFbos);
    glDeleteTextures(3, sentinelTextures);
    glfwDestroyWindow(window); glfwTerminate();
    if (! successStateRestored || ! failureStateRestored || failedTexture != 0
        || a == 0 || b == 0 || hashA == 0 || hashA != hashB
        || legacyTexA == 0 || legacyHashA == 0 || legacyHashA != legacyHashB
        || emptyTex == 0 || emptyHash != 3996078533u || !reactiveValid || !springsValid || !bodiesValid || !geometryValid || !coupledValid || !historyValid || !animatedColliderValid)
    {
        std::cerr << "visual.particles OpenGL determinism/state policy failed: "
                  << hashA << " / " << hashB << " legacy=" << legacyHashA << " / "
                  << legacyHashB << " empty=" << emptyHash
                  << " successState=" << successStateRestored
                  << " failureState=" << failureStateRestored
                  << " timelineReactive=" << reactiveValid
                  << " harmonicSprings=" << springsValid
                  << " collisionBodies=" << bodiesValid
                  << " geometryBinding=" << geometryValid
                  << " coupledBodies=" << coupledValid
                  << " historicalBodies=" << historyValid
                  << " animatedColliders=" << animatedColliderValid << " geometryError=" << geometryReplayError
                  << " first=" << firstLog << " second=" << secondLog << '\n';
        return 1;
    }
    std::cout << "visual.particles OpenGL PASS hash=" << hashA << '\n';
    return 0;
}
