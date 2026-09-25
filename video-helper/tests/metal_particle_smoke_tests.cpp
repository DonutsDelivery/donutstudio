#include "gl_loader.h"
#include "particle_engine.h"
#include "particle_history.h"
#include "particle_geometry_fixture.h"
#include "shader_generator.h"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace
{

struct PixelStats
{
    uint32_t hash = 2166136261u;
    int litPixels = 0;
    uint64_t alphaSum = 0;
};

PixelStats readTexture (arbitgl::GlFuncs& gl, unsigned texture, int width, int height)
{
    unsigned fbo = 0;
    gl.GenFramebuffers (1, &fbo);
    gl.BindFramebuffer (GL_FRAMEBUFFER, fbo);
    gl.FramebufferTexture2D (GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             GL_TEXTURE_2D, texture, 0);
    if (gl.CheckFramebufferStatus (GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
    {
        gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
        gl.DeleteFramebuffers (1, &fbo);
        return {};
    }

    std::vector<uint8_t> pixels (static_cast<size_t> (width) * height * 4);
    glReadPixels (0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
    gl.BindFramebuffer (GL_FRAMEBUFFER, 0);
    gl.DeleteFramebuffers (1, &fbo);

    PixelStats stats;
    for (size_t i = 0; i < pixels.size(); ++i)
    {
        stats.hash ^= pixels[i];
        stats.hash *= 16777619u;
        if ((i & 3u) == 3u)
        {
            stats.alphaSum += pixels[i];
            if (pixels[i] != 0)
                ++stats.litPixels;
        }
    }
    return stats;
}

std::shared_ptr<const canonicalblockc::CanonicalBlockCFrame> oneNote(bool linked = false)
{
    auto score = std::make_shared<arbitmod::Score>();
    score->scoreRevision = 1;
    arbitmod::Note note;
    note.id = 1; note.midiNote = 60; note.velocity = 127.0f;
    note.freqHz = 261.625565f; note.startBeat = 0.0f; note.lengthBeats = 4.0f;
    score->notes.push_back (note);
    if (linked)
    {
        note.id = -102; note.midiNote = 84; note.freqHz *= 4; note.trackId = 1;
        score->notes.push_back(note);
        score->links.push_back({-301, 1, -102, 3, 2, 0});
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

} // namespace

int main()
{
    constexpr int width = 320;
    constexpr int height = 180;

    if (glfwInit() != GLFW_TRUE)
    {
        std::cerr << "GLFW initialization failed; run this test from macOS Terminal\n";
        return 1;
    }
    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
    glfwWindowHint (GLFW_VISIBLE, GLFW_FALSE);
    GLFWwindow* window = glfwCreateWindow (width, height, "Arbit Metal particle smoke",
                                           nullptr, nullptr);
    if (window == nullptr)
    {
        std::cerr << "OpenGL 4.1 context creation failed\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent (window);

    arbitgl::GlFuncs gl;
    std::string missing;
    if (! arbitgl::loadGlFunctions (gl, missing))
    {
        std::cerr << "OpenGL loader failed: " << missing << '\n';
        glfwDestroyWindow (window);
        glfwTerminate();
        return 1;
    }

    videorender::ShaderClock clock;
    clock.frame = 0;
    clock.timeSec = 0.0;
    clock.timeDelta = 1.0 / 30.0;
    clock.playing = true;
    videorender::ParticleParams params;
    params.count = 512;
    params.spawnTrack = 0;
    params.size = 4.0f;
    params.force = 1.0f;
    // Exercise the widened compute/draw uniform ABI, not legacy defaults.
    params.seed = 73;
    params.lifetime = 2.75f;
    params.red = 0.15f;
    params.green = 0.65f;
    params.blue = 0.35f;
    params.alpha = 0.70f;
    const auto notes = oneNote();

    setenv ("ARBIT_PARTICLE_DIAGNOSTICS", "1", 1);
    setenv ("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine fallback;
    unsigned fallbackTexture = 0;
    // Sample while the deterministic upward burst remains inside the viewport.
    // By frame 60 every particle has legitimately travelled above y=1.
    for (int frame = 0; frame <= 15; ++frame)
    {
        clock.frame = frame;
        clock.timeSec = frame * clock.timeDelta;
        fallbackTexture = fallback.render (&gl, clock, width, height, params, notes.get());
    }
    PixelStats fallbackStats;
    if (fallbackTexture != 0)
        fallbackStats = readTexture (gl, fallbackTexture, width, height);
    const auto fallbackDiagnostics = fallback.diagnostics();
    fallback.shutdown (&gl);

    setenv ("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine metal;
    unsigned metalTexture = 0;
    for (int frame = 0; frame <= 15; ++frame)
    {
        clock.frame = frame;
        clock.timeSec = frame * clock.timeDelta;
        metalTexture = metal.render (&gl, clock, width, height, params, notes.get());
    }
    const std::string metalLog = metal.log();
    PixelStats metalStats;
    if (metalTexture != 0)
        metalStats = readTexture (gl, metalTexture, width, height);
    const auto metalDiagnostics = metal.diagnostics();
    metal.shutdown (&gl);

    // A fresh engine rendered directly at frame 15 must replay frames 0..15 and
    // land on the same deterministic image as sequential preview stepping.
    videorender::ParticleEngine metalReplay;
    const unsigned replayTexture = metalReplay.render (&gl, clock, width, height, params, notes.get());
    const PixelStats replayStats = replayTexture != 0
        ? readTexture (gl, replayTexture, width, height) : PixelStats {};
    const auto replayDiagnostics = metalReplay.diagnostics();
    metalReplay.shutdown (&gl);

    params.motionMode = 1;
    params.force = 0.25f;
    params.drag = 0.7f;
    params.attraction = 1.5f;
    params.gravity = 0.1f;
    params.rmsGain = 1.0f; params.rms = 0.25f;
    params.onsetGain = 0.3f; params.onset = 0.5f; params.onsetAge = 0.2f;
    clock.frame = 18; clock.timeSec = 0.6; clock.playing = false;
    videorender::ParticleEngine timelineMetal;
    const auto timelineTexture = timelineMetal.render(&gl, clock, width, height, params, notes.get());
    const auto timelineStats = timelineTexture != 0
        ? readTexture(gl, timelineTexture, width, height) : PixelStats {};
    const bool timelineNative = timelineMetal.log() == "metal";
    clock.frame = 600000; clock.timeSec = 20000.0;
    timelineMetal.render(&gl, clock, width, height, params, notes.get());
    clock.frame = 18; clock.timeSec = 0.6;
    const auto loopTexture = timelineMetal.render(&gl, clock, width, height, params, notes.get());
    const auto loopStats = loopTexture != 0
        ? readTexture(gl, loopTexture, width, height) : PixelStats {};
    timelineMetal.shutdown(&gl);
    setenv("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine timelineFallback;
    const auto timelineGlTexture = timelineFallback.render(&gl, clock, width, height, params, notes.get());
    const auto timelineGlStats = timelineGlTexture != 0
        ? readTexture(gl, timelineGlTexture, width, height) : PixelStats {};
    timelineFallback.shutdown(&gl);

    const auto linkedNotes = oneNote(true);
    params.linkSpring = 2.5f;
    setenv("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine springMetal;
    const auto springTexture = springMetal.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto springStats = springTexture != 0
        ? readTexture(gl, springTexture, width, height) : PixelStats {};
    const bool springNative = springMetal.log() == "metal";
    clock.frame = 600000; clock.timeSec = 20000;
    springMetal.render(&gl, clock, width, height, params, notes.get());
    clock.frame = 18; clock.timeSec = 0.6;
    const auto springLoop = springMetal.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto springLoopStats = springLoop != 0
        ? readTexture(gl, springLoop, width, height) : PixelStats {};
    params.linkRatioInfluence = 0;
    const auto unweightedSpring = springMetal.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto unweightedSpringStats = unweightedSpring != 0
        ? readTexture(gl, unweightedSpring, width, height) : PixelStats {};
    springMetal.shutdown(&gl);
    params.linkRatioInfluence = 1;
    setenv("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine springGl;
    const auto springGlTexture = springGl.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto springGlStats = springGlTexture != 0
        ? readTexture(gl, springGlTexture, width, height) : PixelStats {};
    springGl.shutdown(&gl);

    params.collisionMode = 1; params.bodyRadius = 0.04f;
    params.restitution = 0.9f; params.gravity = 2; params.force = 1;
    clock.timeSec = 2; clock.frame = 60;
    setenv("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine bodyMetal;
    const auto bodyTexture = bodyMetal.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto bodyStats = bodyTexture ? readTexture(gl, bodyTexture, width, height) : PixelStats {};
    const bool bodyNative = bodyMetal.log() == "metal";
    clock.timeSec = 10000; clock.frame = 300000;
    bodyMetal.render(&gl, clock, width, height, params, notes.get());
    clock.timeSec = 2; clock.frame = 60;
    const auto bodyReplay = bodyMetal.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto bodyReplayStats = bodyReplay ? readTexture(gl, bodyReplay, width, height) : PixelStats {};
    bodyMetal.shutdown(&gl);
    setenv("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine bodyGl;
    const auto bodyGlTexture = bodyGl.render(&gl, clock, width, height, params, linkedNotes.get());
    const auto bodyGlStats = bodyGlTexture ? readTexture(gl, bodyGlTexture, width, height) : PixelStats {};
    bodyGl.shutdown(&gl);

    params.collisionMode=0; params.linkSpring=0; params.gravity=0; params.force=0.04f;
    params.geometryCount=2; params.attraction=4;
    params.geometryAnchors[0]={0.25f,0.35f,0.25f,0.35f};
    params.geometryAnchors[1]={0.7f,0.65f,0.7f,0.65f};
    setenv("ARBIT_VIDEO_METAL","1",1);
    videorender::ParticleEngine geometryMetal;
    const auto geometryTexture=geometryMetal.render(&gl,clock,width,height,params,nullptr);
    const auto geometryStats=geometryTexture ? readTexture(gl,geometryTexture,width,height) : PixelStats{};
    const bool geometryNative=geometryMetal.log()=="metal";
    clock.timeSec=10000; clock.frame=300000;
    geometryMetal.render(&gl,clock,width,height,params,nullptr);
    clock.timeSec=2; clock.frame=60;
    const auto geometryReplay=geometryMetal.render(&gl,clock,width,height,params,nullptr);
    const auto geometryReplayStats=geometryReplay ? readTexture(gl,geometryReplay,width,height) : PixelStats{};
    geometryMetal.shutdown(&gl);
    setenv("ARBIT_VIDEO_METAL","0",1);
    videorender::ParticleEngine geometryGl;
    const auto geometryGlTexture=geometryGl.render(&gl,clock,width,height,params,nullptr);
    const auto geometryGlStats=geometryGlTexture ? readTexture(gl,geometryGlTexture,width,height) : PixelStats{};
    geometryGl.shutdown(&gl);

    params = {};
    params.motionMode = 2; params.count = 2; params.geometryCount = 2;
    params.lifetime = 4; params.force = 0; params.bodyRadius = 0.07f;
    params.collisionMode = 2; params.bodyShape = 1; params.size = 12; params.impulseY = 0.1f;
    params.geometryAnchors[0] = {0.48f,0.5f,0.48f,0.5f};
    params.geometryAnchors[1] = {0.52f,0.54f,0.52f,0.54f};
    clock.timeSec = 0.5; clock.frame = 15;
    setenv("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine coupledMetal;
    const auto coupledTexture = coupledMetal.render(&gl, clock, width, height, params, nullptr);
    const auto coupledStats = coupledTexture ? readTexture(gl, coupledTexture, width, height) : PixelStats{};
    const bool coupledNative = coupledMetal.log() == "metal";
    clock.timeSec = 10000; clock.frame = 300000;
    coupledMetal.render(&gl, clock, width, height, params, nullptr);
    clock.timeSec = 0.5; clock.frame = 15;
    const auto coupledReplay = coupledMetal.render(&gl, clock, width, height, params, nullptr);
    const auto coupledReplayStats = coupledReplay ? readTexture(gl, coupledReplay, width, height) : PixelStats{};
    coupledMetal.shutdown(&gl);
    setenv("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine coupledGl;
    const auto coupledGlTexture = coupledGl.render(&gl, clock, width, height, params, nullptr);
    const auto coupledGlStats = coupledGlTexture ? readTexture(gl, coupledGlTexture, width, height) : PixelStats{};
    coupledGl.shutdown(&gl);

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
    params.historicalReplay = true; params.history = history;
    params.rmsGain = 0.1f; params.onsetGain = 0.2f;
    params.historyProjectSeconds = clock.timeSec;
    setenv("ARBIT_VIDEO_METAL", "1", 1);
    videorender::ParticleEngine historyMetal;
    const auto historyTexture = historyMetal.render(&gl, clock, width, height, params, nullptr);
    const auto historyStats = historyTexture ? readTexture(gl, historyTexture, width, height) : PixelStats{};
    const bool historyNative = historyMetal.log() == "metal";
    clock.timeSec = 10000; clock.frame = 300000; params.historyProjectSeconds = clock.timeSec;
    historyMetal.render(&gl, clock, width, height, params, nullptr);
    clock.timeSec = 0.5; clock.frame = 15; params.historyProjectSeconds = clock.timeSec;
    const auto historyReplay = historyMetal.render(&gl, clock, width, height, params, nullptr);
    const auto historyReplayStats = historyReplay ? readTexture(gl, historyReplay, width, height) : PixelStats{};
    historyMetal.shutdown(&gl);
    setenv("ARBIT_VIDEO_METAL", "0", 1);
    videorender::ParticleEngine historyGl;
    const auto historyGlTexture = historyGl.render(&gl, clock, width, height, params, nullptr);
    const auto historyGlStats = historyGlTexture ? readTexture(gl, historyGlTexture, width, height) : PixelStats{};
    historyGl.shutdown(&gl);
    std::string colliderError;
    auto collider=particleGeometryFixture(true,colliderError);
    bool colliderValid=collider && prepareParticleGeometryFixture(*collider,0.5,colliderError);
    if (colliderValid)
    {
        setenv("ARBIT_VIDEO_METAL","1",1);
        videorender::ParticleEngine nativeCollider;
        const auto texture=nativeCollider.render(&gl,clock,width,height,*collider,nullptr);
        const auto native=texture ? readTexture(gl,texture,width,height) : PixelStats{};
        colliderValid=nativeCollider.log()=="metal" && native.litPixels>0;
        nativeCollider.shutdown(&gl);
        setenv("ARBIT_VIDEO_METAL","0",1);
        videorender::ParticleEngine glCollider;
        const auto fallbackTexture=glCollider.render(&gl,clock,width,height,*collider,nullptr);
        const auto fallback=fallbackTexture ? readTexture(gl,fallbackTexture,width,height) : PixelStats{};
        colliderValid=colliderValid && fallback.litPixels>0
            && native.litPixels>=fallback.litPixels/2 && native.litPixels<=fallback.litPixels*2;
        glCollider.shutdown(&gl);
    }

    glfwDestroyWindow (window);
    glfwTerminate();

    std::cout << "GL fallback: hash=" << fallbackStats.hash
              << " litPixels=" << fallbackStats.litPixels
              << " alphaSum=" << fallbackStats.alphaSum << '\n';
    std::cout << "GL stages: notes=" << fallbackDiagnostics.noteRows
              << " live=" << fallbackDiagnostics.liveParticles
              << " visible=" << fallbackDiagnostics.visibleCandidates
              << " drawAlpha=" << fallbackDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << fallbackDiagnostics.readbackAlphaPixels << '\n';
    std::cout << "Metal bridge: hash=" << metalStats.hash
              << " litPixels=" << metalStats.litPixels
              << " alphaSum=" << metalStats.alphaSum
              << " backend=" << metalLog << '\n';
    std::cout << "Metal stages: notes=" << metalDiagnostics.noteRows
              << " live=" << metalDiagnostics.liveParticles
              << " visible=" << metalDiagnostics.visibleCandidates
              << " drawAlpha=" << metalDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << metalDiagnostics.readbackAlphaPixels << '\n';
    std::cout << "Metal direct replay: hash=" << replayStats.hash
              << " live=" << replayDiagnostics.liveParticles
              << " drawAlpha=" << replayDiagnostics.drawAlphaPixels
              << " readbackAlpha=" << replayDiagnostics.readbackAlphaPixels << '\n';

    if (fallbackTexture == 0 || fallbackStats.litPixels == 0
        || fallbackDiagnostics.noteRows != 1 || fallbackDiagnostics.liveParticles <= 0
        || fallbackDiagnostics.drawAlphaPixels <= 0)
    {
        std::cerr << "OpenGL reference path produced no particles\n";
        return 1;
    }
    if (metalTexture == 0 || metalLog != "metal" || metalStats.litPixels == 0
        || metalDiagnostics.noteRows != 1 || metalDiagnostics.liveParticles <= 0
        || metalDiagnostics.drawAlphaPixels <= 0
        || metalDiagnostics.readbackAlphaPixels <= 0
        || replayTexture == 0 || replayStats.hash != metalStats.hash
        || replayDiagnostics.liveParticles != metalDiagnostics.liveParticles
        || replayDiagnostics.drawAlphaPixels <= 0
        || replayDiagnostics.readbackAlphaPixels <= 0)
    {
        std::cerr << "Metal particle/IOSurface bridge failed: " << metalLog << '\n';
        return 1;
    }
    if (!timelineNative || timelineStats.litPixels == 0 || timelineGlStats.litPixels == 0
        || loopStats.hash != timelineStats.hash)
    {
        std::cerr << "Timeline particle pause/seek/loop native parity failed\n";
        return 1;
    }
    if (!springNative || springStats.litPixels == 0 || springGlStats.litPixels == 0
        || springStats.hash == timelineStats.hash || springLoopStats.hash != springStats.hash
        || unweightedSpringStats.hash == springStats.hash)
    {
        std::cerr << "Harmonic particle spring endpoint/ratio/loop parity failed\n";
        return 1;
    }
    if (!bodyNative || bodyStats.litPixels == 0 || bodyGlStats.litPixels == 0
        || bodyReplayStats.hash != bodyStats.hash)
    {
        std::cerr << "Bounded particle body native replay failed\n";
        return 1;
    }
    if (!geometryNative || geometryStats.litPixels==0 || geometryGlStats.litPixels==0
        || geometryReplayStats.hash!=geometryStats.hash
        || geometryStats.litPixels<geometryGlStats.litPixels/2
        || geometryStats.litPixels>geometryGlStats.litPixels*2) {
        std::cerr << "Geometry particle emitter/attractor native replay failed\n";
        return 1;
    }
    if (!coupledNative || coupledStats.litPixels == 0 || coupledGlStats.litPixels == 0
        || coupledReplayStats.hash != coupledStats.hash
        || coupledStats.litPixels < coupledGlStats.litPixels / 2
        || coupledStats.litPixels > coupledGlStats.litPixels * 2)
    {
        std::cerr << "Coupled snapshot body Metal/GL replay failed\n";
        return 1;
    }
    if (!colliderValid || !historyNative || historyStats.litPixels == 0 || historyGlStats.litPixels == 0
        || historyReplayStats.hash != historyStats.hash
        || historyStats.litPixels < historyGlStats.litPixels / 2
        || historyStats.litPixels > historyGlStats.litPixels * 2)
    {
        std::cerr << "Historical body Metal/GL replay failed\n";
        return 1;
    }
    const double springCoverage = static_cast<double>(springStats.litPixels) / springGlStats.litPixels;
    if (springCoverage < 0.5 || springCoverage > 2.0)
    {
        std::cerr << "Harmonic particle spring Metal/GL coverage differs\n";
        return 1;
    }
    const double timelineRatio = static_cast<double>(timelineStats.litPixels)
        / static_cast<double>(timelineGlStats.litPixels);
    if (timelineRatio < 0.5 || timelineRatio > 2.0)
    {
        std::cerr << "Timeline particle Metal/GL coverage differs\n";
        return 1;
    }

    const double litRatio = static_cast<double> (metalStats.litPixels)
                          / static_cast<double> (fallbackStats.litPixels);
    if (litRatio < 0.5 || litRatio > 2.0)
    {
        std::cerr << "Metal/GL particle coverage differs unexpectedly (ratio "
                  << litRatio << ")\n";
        return 1;
    }
    std::cout << "Metal particle bridge smoke passed; coverage ratio="
              << litRatio << '\n';
    return 0;
}
