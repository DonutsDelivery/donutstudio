// An edit cancels obsolete render work (Plan 3, export and encoders). The RPCs
// that change a frame's inputs bump the render-input generation; a preview pass
// that captured an older generation stops at its sample-loop poll, and says so;
// an export never watches it.
// Each clip has its own generation: an edit to one clip leaves another clip's
// preview converging. Renders stay at 16 x 16 and 2 samples per pixel.

#include "pathtrace/engine.h"
#include "pathtrace/thread_pool.h"
#include "render_input_generation.h"
#include "support/pathtrace_glass_scene.h"

#include <iostream>
#include <string>

namespace
{

namespace pt = videohelper::pathtrace;
namespace renderinput = videohelper::renderinput;
namespace glass = pathtraceglassscene;

int checks = 0;
int failures = 0;

void check (bool condition, const std::string& message)
{
    ++checks;
    if (! condition)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

void methodTests()
{
    for (const char* method : { "graph_set_param", "graph_set_effects", "graph_set_lut", "text_set_image",
                                "viewport_set_timeline", "viewport_set_post", "viewport_set_mod_matrix" })
        check (renderinput::changesFrameInputs (method), std::string (method) + " changes a frame's inputs");
    for (const char* method : { "viewport_set_visible", "viewport_set_bounds", "graph_describe", "export",
                                "pathtrace_device", "" })
        check (! renderinput::changesFrameInputs (method), std::string ("'") + method + "' leaves them alone");

    const auto before = renderinput::current();
    check (renderinput::bump() == before + 1 && renderinput::current() == before + 1, "a bump moves the generation by one");
}

void tokenTests()
{
    pt::CancellationToken token;
    check (! token.isCancelled() && ! token.obsolete(), "a fresh token runs");
    const auto captured = renderinput::current();
    token.watch (&renderinput::counter(), captured);
    check (! token.isCancelled(), "a pass on the current inputs runs");
    renderinput::bump();
    check (token.isCancelled() && token.obsolete(), "an edit after capture stops the pass");
    token.reset();
    check (token.isCancelled(), "reset re-arms the flag, not the obsolete inputs");
    token.watch (&renderinput::counter(), renderinput::current());
    check (! token.isCancelled(), "a pass that captured the edit runs");
    token.cancel();
    check (token.isCancelled() && ! token.obsolete(), "an explicit cancel is not an edit");
    token.reset();
    token.watch (nullptr, 0);
    renderinput::bump();
    check (! token.isCancelled(), "a token that watches nothing ignores edits");
}

void engineTests()
{
    const auto scene = glass::makeScene();
    const auto setup = glass::makeSetup (scene);
    const auto view = glass::makeView (*scene, 16, 16);
    std::string error;
    pt::SceneEngine engine (1);
    engine.maximumPreviewSamples = 2;
    engine.samplesPerPass = 1;

    pt::EngineImage image;
    engine.watchInputs (&renderinput::counter(), renderinput::current());
    check (engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error), "a current preview renders: " + error);
    check (image.traced && ! image.stats.cancelled && ! engine.lastPassObsolete(), "and completes its pass");

    // The pass captured the inputs before an edit landed: it stops in its loop.
    engine.watchInputs (&renderinput::counter(), renderinput::current());
    renderinput::bump();
    check (engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error),
           "an obsolete preview is not an error: " + error);
    check (image.stats.cancelled && engine.lastPassObsolete(), "an obsolete preview pass stops and says why");
    check (image.stats.samplesAdded < 16u * 16u, "it stops before tracing every pixel");

    engine.watchInputs (&renderinput::counter(), renderinput::current());
    check (engine.renderPreview (setup, view, pt::EngineColor::LinearRgba16f, image, error), "the next frame renders");
    check (image.restarted && ! image.stats.cancelled && ! engine.lastPassObsolete(), "and restarts on the edited inputs");

    // An export renders its snapshot to the end, whatever the watch said.
    engine.watchInputs (&renderinput::counter(), renderinput::current());
    renderinput::bump();
    pt::EngineImage exported;
    check (engine.renderExport (setup, view, pt::EngineColor::LinearRgba16f, 2, exported, error),
           "an export ignores edits: " + error);
    check (! exported.stats.cancelled && exported.samplesPerPixel == 2, "and adds every sample");
}

void perClipTests()
{
    using Kind = renderinput::EditScope::Kind;
    check (renderinput::clipOfParamId ("clip7/visual3/roughness") == 7, "a clip parameter names its clip");
    check (! renderinput::clipOfParamId ("clip/x").has_value() && ! renderinput::clipOfParamId ("clip0/x").has_value()
               && ! renderinput::clipOfParamId ("clip12").has_value() && ! renderinput::clipOfParamId ("text3/size").has_value(),
           "malformed and non-clip ids name no clip");

    auto scope = renderinput::editScope ("graph_set_param", "clip7/visual3/roughness", std::nullopt);
    check (scope.kind == Kind::Clip && scope.clip == 7, "a parameter edit reaches its clip only");
    check (renderinput::editScope ("graph_set_param", "view/zoom", std::nullopt).kind == Kind::None,
           "a view parameter changes no frame");
    check (renderinput::editScope ("graph_set_param", "text4/size", std::nullopt).kind == Kind::All,
           "a text parameter reaches every clip");
    scope = renderinput::editScope ("graph_set_effects", "", 9);
    check (scope.kind == Kind::Clip && scope.clip == 9, "an effect edit reaches its clip");
    scope = renderinput::editScope ("graph_set_lut", "", 9);
    check (scope.kind == Kind::Clip && scope.clip == 9, "a LUT edit reaches its clip");
    check (renderinput::editScope ("viewport_set_timeline", "", std::nullopt).kind == Kind::All, "the timeline reaches every clip");
    check (renderinput::editScope ("transport_play", "clip7/a/b", 7).kind == Kind::None, "transport changes no frame");

    check (&renderinput::counter (3) == &renderinput::counter (3), "a clip keeps one generation");
    check (&renderinput::counter (0) == &renderinput::counter(), "clip 0 is the shared generation");

    pt::CancellationToken clip3, clip4, shared;
    clip3.watch (&renderinput::counter (3), renderinput::current (3));
    clip4.watch (&renderinput::counter (4), renderinput::current (4));
    shared.watch (&renderinput::counter(), renderinput::current());
    renderinput::bump (renderinput::editScope ("graph_set_param", "clip3/visual1/exposure", std::nullopt));
    check (clip3.obsolete(), "an edit to clip 3 stops clip 3's preview");
    check (! clip4.obsolete(), "and leaves clip 4's preview converging");
    check (shared.obsolete(), "and stops a draw outside any clip scope");

    clip3.watch (&renderinput::counter (3), renderinput::current (3));
    renderinput::bump (renderinput::editScope ("viewport_set_canvas", "", std::nullopt));
    check (clip3.obsolete() && clip4.obsolete(), "a canvas edit stops every clip's preview");

    clip4.watch (&renderinput::counter (4), renderinput::current (4));
    renderinput::bump (renderinput::editScope ("graph_set_param", "view/panX", std::nullopt));
    check (! clip4.obsolete(), "a view pan stops nothing");

    check (renderinput::activeClip() == 0, "no clip is active outside a layer");
    {
        const renderinput::ClipScope outer (5);
        check (renderinput::activeClip() == 5, "a layer scope names its clip");
        {
            const renderinput::ClipScope inner (6);
            check (renderinput::activeClip() == 6, "a nested layer names its own");
        }
        check (renderinput::activeClip() == 5, "and gives the outer clip back");
    }
    check (renderinput::activeClip() == 0, "the scope ends with the layer");
}

} // namespace

int main()
{
    methodTests();
    tokenTests();
    engineTests();
    perClipTests();
    std::cout << (checks - failures) << "/" << checks << " render input generation checks passed\n";
    return failures == 0 ? 0 : 1;
}
