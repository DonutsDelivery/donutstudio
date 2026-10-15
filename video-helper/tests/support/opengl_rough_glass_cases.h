#pragma once
#include "smooth_glass_cases.h"
#include "glass_caustic_oracle.h"
#include "../../src/gpu_backend/fixture_rough_glass_tiles_opengl.h"
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

namespace arbitgpu
{
// Implementation belongs to the GL backend; receipts remain private frame
// scratch and this read-only native diagnostic has no wire or AOV selector.
bool readOpenGlRoughGlassReceiptForFixture(const NativeFixtureSceneFrame&,
    std::array<NativeRawPassPixels, 3>&, NativeRawPassPixels&, NativeRawPassPixels&, std::string&);
}

namespace videohelper::tests::openglroughglassfixture
{
using namespace HarmonicMIDI::grid;
using smoothglassfixture::D3;
inline bool selectedShaderAssembly(std::string& error)
{
    // Inspect the exact source compiled by the native adapter, including both query namespaces.
    for (bool transport : {false,true})
    {
        const auto source = arbitgpu::openGlSelectedRoughCameraShaderForFixture(transport);
        if (source.find("#version 430 core\n") != 0)
        { error = "GL selected camera shader lost its compute version"; return false; }
        for (const auto* macro : {"SC_TEXELS","ST_READ"})
        {
            const auto definition = std::string("#define ") + macro + " ";
            const auto reset = source.find(std::string("#undef ") + macro + "\n");
            const auto first = source.find(definition);
            const auto second = first == std::string::npos ? first : source.find(definition,first+definition.size());
            if (reset == std::string::npos || first == std::string::npos
                || (transport ? !(first < reset && reset < second && second != std::string::npos
                    && source.find(definition,second+definition.size()) == std::string::npos)
                    : !(reset < first && second == std::string::npos)))
            { error = "GL selected camera query macro lacks its explicit namespace reset: " + std::string(macro); return false; }
        }
        const auto opaque = source.find("uniform sampler2D uOpaqueVisibility;");
        const auto glass = source.find("uniform sampler2D uGlassVisibility;");
        if (glass == std::string::npos || (transport ? !(opaque < glass) : opaque != std::string::npos))
        { error = "GL selected camera query namespace order changed"; return false; }
    }
    return true;
}
inline float scalar(const arbitgpu::NativeRawPassPixels& pixels, std::size_t pixel, std::size_t channel)
{
    float value = 0;
    std::memcpy(&value, pixels.bytes.data() + pixel * 16 + channel * 4, 4);
    return value;
}
inline std::uint32_t identity(const arbitgpu::NativeRawPassPixels& pixels, std::size_t pixel)
{
    std::uint32_t value = 0;
    std::memcpy(&value, pixels.bytes.data() + pixel * 4, 4);
    return value;
}
inline void write(arbitgpu::NativeRawPassPixels& pixels, std::size_t pixel, std::size_t channel, float value)
{
    std::memcpy(pixels.bytes.data() + pixel * 16 + channel * 4, &value, 4);
}
inline std::shared_ptr<const videorender::fixture3d::AdmittedSurfaceMaterialBinding> binding(
    const std::shared_ptr<const Visual3DScene>& authored, float roughness, float ior, float dispersion,
    std::string& error)
{
    surfacematerialbinding::ImportedSceneMaterialRequest request;
    request.scene = authored->id;
    request.sceneRevision = request.structuralRevision = request.evaluationRevision = request.programRevision = 1;
    request.sceneSnapshot = authored;
    request.program = smoothglassfixture::authoredProgram(6, ior, dispersion);
    for (auto& operation : request.program.operations)
        if (operation.id == static_cast<std::uint32_t>(surfacematerial::OutputSemantic::Roughness) + 1)
            operation.literal[0] = roughness;
    request.binding.targetKind = surfacematerialbinding::BindingTargetKind::ObjectOverride;
    request.binding.object = authored->objects[0].id;
    request.binding.surfaceMaterialRevision = 1;
    const auto admitted = surfacematerial::admit(request.program, error);
    if (!admitted) return {};
    request.binding.surfaceMaterialDigest = admitted->structuralDigest();
    return videorender::fixture3d::admitSurfaceMaterialBinding(authored, request,
        videohelper::materialprogram::BackendTarget::OpenGl, error);
}
inline bool malformed(const arbitgpu::smoothglass::Prepared& prepared,
    const std::array<arbitgpu::NativeRawPassPixels, 3>& native, const arbitgpu::NativeRawPassPixels& objects,
    const arbitgpu::NativeRawPassPixels& materials, std::size_t covered, std::string& error)
{
    // Feed corrupted copies of an actual native receipt through the exact
    // validator used before native publication; never mutate a retained frame.
    for (int kind = 0; kind < 9; ++kind)
    {
        auto receipt = native;
        switch (kind)
        {
            case 0: receipt[0].bytes.pop_back(); break;
            case 1: ++receipt[2].width; break;
            case 2: write(receipt[1], covered, 0, std::numeric_limits<float>::quiet_NaN()); break;
            case 3: write(receipt[0], covered, 0, .5f); break;
            case 4:
                write(receipt[0], covered, 0, .99f);
                write(receipt[1], covered, 0, .01f);
                write(receipt[2], covered, 0, 0); break;
            case 5: write(receipt[0], covered, 3, 0); break;
            case 6: write(receipt[1], covered, 3, 1.5f); break;
            case 7:
                write(receipt[0], covered, 3, -2);
                write(receipt[2], covered, 3, -2); break;
            case 8:
                for (auto& image : receipt) for (std::size_t channel = 0; channel < 4; ++channel)
                    write(image,covered,channel,0);
                break;
        }
        arbitgpu::NativeFixtureSceneStats stats;
        stats.roughGlassCoveredPixels = 987;
        if (arbitgpu::smoothglass::completeRoughFrame(prepared, objects.width, objects.height,
                receipt[0], receipt[1], receipt[2], objects, materials, stats, error)
            || stats.roughGlassReceiptComplete || stats.roughGlassCoveredPixels != 987)
        { error = "GL malformed/stale rough receipt admitted or published partial stats"; return false; }
        if (kind == 7 && error.find("Internal") == std::string::npos && error.find("internal") == std::string::npos)
        { error = "GL rough internal-camera status was confused with legacy Color overflow"; return false; }
    }
    error.clear();
    return true;
}
inline bool run(arbitgpu::NativeFixtureSceneBackend& backend, std::string& error)
{
    if (!selectedShaderAssembly(error)) return false;
    constexpr std::uint32_t extent = 9;
    struct Case { float roughness, dispersion, receiverZ; bool blocker, foreground; };
    const Case cases[] {{.04f,.004f,-3,false,false}, {.08f,.004f,-3,false,false},
        {.08f,0,-3,false,false}, {.08f,.004f,-4,false,false},
        {.08f,.004f,-3,true,false}, {.08f,.004f,-3,false,true}};
    auto footprintScene = smoothglassfixture::scene(.25f,1,20);
    smoothglassfixture::receiver(footprintScene,-6,{1,1,1});
    footprintScene.lights[0].color = {};
    const double angle = 20 * std::acos(-1.0) / 180;
    const D3 camera { footprintScene.cameras[0].transform.translation.x,0,
        footprintScene.cameras[0].transform.translation.z };
    const D3 centreRay { -std::sin(angle),0,-std::cos(angle) };
    const auto narrow = smoothglassfixture::traceRough(footprintScene,camera,centreRay,1.5,.004,.04);
    const auto broad = smoothglassfixture::traceRough(footprintScene,camera,centreRay,1.5,.004,.08);
    if (!narrow.optics.complete || !broad.optics.complete || narrow.footprintVariance[1] <= 0
        || broad.footprintVariance[1] <= narrow.footprintVariance[1] * 2)
    { error = "GL rough fixture lacks an independently broadened triangle receiver footprint"; return false; }
    const float boundary = static_cast<float>(narrow.footprintMean[1]);
    std::array<std::vector<std::uint8_t>, 6> nativeColors;
    bool observedNonzeroTail = false, observedOcclusion = false;
    std::size_t row = 0;
    for (const auto& test : cases)
    {
        auto authored = smoothglassfixture::scene(.25f,1,20);
        authored.lights[0].color = {};
        smoothglassfixture::receiver(authored,boundary,{1,1,1});
        for (std::size_t v = 8; v < 12; ++v) authored.vertices[v].position.z = test.receiverZ;
        if (test.blocker || test.foreground)
        {
            const float edge = test.foreground
                ? static_cast<float>(camera[0]+(2-camera[2])*centreRay[0]/centreRay[2]) : -6;
            smoothglassfixture::receiver(authored,edge,{});
            for (std::size_t v = 12; v < 16; ++v)
                authored.vertices[v].position.z = test.foreground ? 2 : -2;
        }
        auto snapshot = std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material = binding(snapshot,test.roughness,1.5f,test.dispersion,error);
        if (!material) return false;
        videorender::fixture3d::FixtureSceneRenderer renderer(backend);
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.linearColor = true;
        videorender::fixture3d::RenderedFrame preview, exported;
        if (!renderer.renderPreview(snapshot,material,{},inputs,{extent,extent},"native-gpu",preview,error)
            || !renderer.renderExport(snapshot,material,{},inputs,{extent,extent},"native-gpu",exported,error))
        { error = "GL native rough preview/export refused fixture row " + std::to_string(row) + ": " + error; return false; }
        const auto& stats = preview.stats;
        if (!stats.roughGlassReceiptComplete || stats.roughGlassCoveredPixels == 0
            || stats.roughGlassReceiptBytes != extent*extent*48
            || stats.roughGlassMaximumEnergyResidual > .00002
            || stats.roughGlassMaximumUnresolvedPower > .0001)
        { error = "GL native rough frame omitted its complete power/resource receipt"; return false; }
        std::array<arbitgpu::NativeRawPassPixels,3> receipt, exportReceipt;
        arbitgpu::NativeRawPassPixels objects, materials, exportObjects, exportMaterials, color, exportColor, normal;
        if (!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*preview.nativeFrame,receipt,objects,materials,error)
            || !arbitgpu::readOpenGlRoughGlassReceiptForFixture(*exported.nativeFrame,exportReceipt,exportObjects,exportMaterials,error)
            || !preview.nativeFrame->readRawPass(renderpassoutput::Output::Color,color,error)
            || !exported.nativeFrame->readRawPass(renderpassoutput::Output::Color,exportColor,error)
            || !preview.nativeFrame->readRawPass(renderpassoutput::Output::Normal,normal,error)) return false;
        if (color.bytes != exportColor.bytes || objects.bytes != exportObjects.bytes || materials.bytes != exportMaterials.bytes)
        { error = "GL rough native preview/export Color or visibility identity differs"; return false; }
        for (std::size_t i = 0; i < receipt.size(); ++i)
            if (receipt[i].format != arbitgpu::NativeTexturePixelFormat::Rgba32Float
                || receipt[i].bytes != exportReceipt[i].bytes)
            { error = "GL rough native preview/export float32 receipt identity differs"; return false; }
        std::size_t coveredCount = 0, firstCovered = extent*extent;
        const double tangent = std::tan(snapshot->cameras[0].verticalFovRadians*.5);
        for (std::size_t y = 0; y < extent; ++y) for (std::size_t x = 0; x < extent; ++x)
        {
            const std::size_t pixel = y*extent+x;
            const double horizontal = (2*(double(x)+.5)/extent-1)*tangent;
            const double vertical = (1-2*(double(y)+.5)/extent)*tangent;
            const D3 ray = smoothglassfixture::unit({std::cos(angle)*horizontal-std::sin(angle),
                vertical,-std::sin(angle)*horizontal-std::cos(angle)});
            const auto first = smoothglassfixture::triangleHit(*snapshot,camera,ray);
            if (!first.hit || identity(objects,pixel) != first.object || identity(materials,pixel) != first.material)
            { error = "GL rough receipt raw IDs differ from independent primary triangle visibility"; return false; }
            if (first.object != snapshot->objects[0].id.value)
            {
                observedOcclusion = true;
                for (const auto& image : receipt) for (std::size_t c = 0; c < 4; ++c)
                    if (scalar(image,pixel,c) != 0)
                    { error = "GL rough receipt wrote power through opaque foreground occlusion"; return false; }
                continue;
            }
            ++coveredCount;
            firstCovered = std::min(firstCovered,pixel);
            const auto expected = smoothglassfixture::traceRough(*snapshot,camera,ray,1.5,test.dispersion,test.roughness);
            if (!expected.optics.complete)
            { error = "GL independent rough triangle oracle did not complete a visible pixel"; return false; }
            if (scalar(receipt[0],pixel,3) != 1 || scalar(receipt[2],pixel,3) != 1
                || scalar(receipt[1],pixel,3) != expected.closestHitQueries)
            { error = "GL rough native completion/query receipt differs from the independent schedule"; return false; }
            for (std::size_t c = 0; c < 3; ++c)
            {
                // Power is checked in float32 independently of binary16 Color.
                const double actual[] {scalar(receipt[0],pixel,c),scalar(receipt[1],pixel,c),scalar(receipt[2],pixel,c)};
                const double powers[] {expected.energy.terminal[c],expected.energy.masked[c],expected.energy.tail[c]};
                for (std::size_t component = 0; component < 3; ++component)
                    if (!std::isfinite(actual[component]) || std::abs(actual[component]-powers[component]) > .000004)
                    { error = "GL rough native per-pixel power differs from the binary64 triangle oracle"; return false; }
                observedNonzeroTail |= actual[2] > 0 && powers[2] > 0;
                std::uint16_t half = 0;
                std::memcpy(&half,color.bytes.data()+pixel*8+c*2,2);
                const double radiance = hdrimage::halfToFloat(half);
                if (!std::isfinite(radiance) || std::abs(radiance-expected.optics.rgb[c]) > .0015)
                { error = "GL rough native receiver footprint/band radiance differs from the independent triangle oracle"; return false; }
            }
            std::uint16_t marker = 0;
            std::memcpy(&marker,normal.bytes.data()+pixel*8+6,2);
            if (marker != 0x3c00u)
            { error = "GL rough completion changed the legacy public Normal alpha semantics"; return false; }
        }
        if (coveredCount != stats.roughGlassCoveredPixels)
        { error = "GL rough receipt coverage count differs from raw visibility"; return false; }
        if (row == 0)
        {
            arbitgpu::smoothglass::Prepared prepared;
            if (!arbitgpu::smoothglass::prepare(snapshot,material->nativeProgram(),{},prepared,error,
                    {arbitgpu::NativeFixtureMaterialBackend::OpenGl,true,true,true})
                || !malformed(prepared,receipt,objects,materials,firstCovered,error)) return false;
            auto nativePreparation = backend.prepare(snapshot,material->nativeProgram());
            if (!nativePreparation.prepared) { error = nativePreparation.error; return false; }
            const auto resized = backend.render(snapshot,nativePreparation.resources,5,7,inputs);
            if (!resized.rendered || !resized.frame || resized.stats.textureBytes != 5*7*48
                || resized.stats.roughGlassReceiptBytes != 5*7*48)
            { error = "GL rough resize omitted its actual frame scratch allocation charge"; return false; }
            std::array<arbitgpu::NativeRawPassPixels,3> retained;
            arbitgpu::NativeRawPassPixels retainedObjects, retainedMaterials;
            if (!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*preview.nativeFrame,retained,
                    retainedObjects,retainedMaterials,error)) return false;
            for (std::size_t i = 0; i < receipt.size(); ++i)
                if (retained[i].bytes != receipt[i].bytes || retained[i].width != extent || retained[i].height != extent)
                { error = "GL rough resize reused or retired an older frame's private receipt"; return false; }
            auto privateSelector = inputs;
            privateSelector.imageOutput = static_cast<renderpassoutput::Output>(255);
            const auto invalidSelector = backend.render(snapshot,nativePreparation.resources,extent,extent,privateSelector);
            if (invalidSelector.rendered || invalidSelector.frame)
            { error = "GL private receipt selector became a public image output"; return false; }
            const auto tiled = backend.render(snapshot,nativePreparation.resources,100,100,inputs);
            if (!tiled.rendered || !tiled.frame || !tiled.stats.roughGlassReceiptComplete)
            { error = "GL rough bounded tile jobs refused a completed 100x100 frame: " + tiled.error; return false; }
            std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> schedule;
            arbitgpu::roughglasstiles::Aggregate aggregate;
            if (!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*tiled.frame,schedule,aggregate,error)
                || schedule->tiles().size()<2 || aggregate.completedPixels!=10000
                || aggregate.completedTiles!=schedule->tiles().size()
                || aggregate.reservedQueries!=schedule->reservedQueries()
                || aggregate.peakReceiptBytes!=tiled.stats.roughGlassReceiptBytes
                || aggregate.peakReceiptBytes>9708*48)
            { error = "GL rough full frame omitted its bounded complete native tile summary"; return false; }
            // Multi-job scratch contains only the last job. Never expose it as
            // a fabricated full-frame receipt or assemble product pixels on CPU.
            if (arbitgpu::readOpenGlRoughGlassReceiptForFixture(*tiled.frame,retained,
                retainedObjects,retainedMaterials,error))
            { error = "GL rough tiled scratch was mislabeled as a full-frame receipt"; return false; }
            error.clear();
            arbitgpu::NativeRawPassPixels tiledColor, tiledObjects;
            if (!tiled.frame->readRawPass(renderpassoutput::Output::Color,tiledColor,error)
                || !tiled.frame->readRawPass(renderpassoutput::Output::ObjectId,tiledObjects,error)) return false;
            // Exercise first/last pixels and both sides of every tile edge with
            // full-frame camera rays. The 9x9 cases above remain the exact small
            // baseline; this checks the same estimator at translated scratch.
            std::vector<std::size_t> edgePixels;
            std::vector<bool> covered(10000,false);
            std::uint32_t jobIndex=0;
            for (const auto& tile : schedule->tiles()) {
                if (tile.index!=jobIndex++) {error="GL rough native jobs changed deterministic order";return false;}
                for (std::uint32_t y=tile.y;y<tile.y+tile.height;++y)
                    for (std::uint32_t x=tile.x;x<tile.x+tile.width;++x) {
                        if (x>=100 || y>=100 || covered[y*100+x])
                        {error="GL rough native jobs overlap or leave their immutable full-frame bounds";return false;}
                        covered[y*100+x]=true;
                    }
                if (tile.pixels()>9708 || tile.reservedQueries()>arbitgpu::smoothglass::kMaximumPixelRayQueries)
                { error = "GL rough native job exceeded its existing query authority"; return false; }
                for (const auto y : {tile.y,tile.y+tile.height-1})
                    for (const auto x : {tile.x,tile.x+tile.width-1}) edgePixels.push_back(y*100+x);
            }
            if (std::find(covered.begin(),covered.end(),false)!=covered.end())
            {error="GL rough native tile edges leave an unpublished full-frame hole";return false;}
            arbitgl::GlFuncs diagnosticGl;std::string missingGl;
            if (!arbitgl::loadGlFunctions(diagnosticGl,missingGl))
            {error="GL rough readback diagnostic is missing native entry points: "+missingGl;return false;}
            unsigned incompleteFramebuffer=0;diagnosticGl.GenFramebuffers(1,&incompleteFramebuffer);
            arbitgpu::NativeRawPassPixels refusedReadback;
            const bool readbackAccepted=arbitgpu::roughglasstiles::opengl::readRegion(diagnosticGl,
                incompleteFramebuffer,GL_COLOR_ATTACHMENT0,0,0,1,1,
                arbitgpu::NativeTexturePixelFormat::Rgba32Float,refusedReadback,error);
            diagnosticGl.DeleteFramebuffers(1,&incompleteFramebuffer);
            if (readbackAccepted || !refusedReadback.bytes.empty())
            {error="GL rough tile incomplete native framebuffer produced an owned readback";return false;}
            if (arbitgpu::roughglasstiles::opengl::readRegion(diagnosticGl,0,GL_COLOR_ATTACHMENT0,0,0,
                std::numeric_limits<std::uint32_t>::max(),2,arbitgpu::NativeTexturePixelFormat::Rgba32Float,refusedReadback,error))
            {error="GL rough native region admitted an overflow-sized tile readback";return false;}
            error.clear();
            for (const auto pixel : edgePixels) {
                const auto x=pixel%100,y=pixel/100;
                const double horizontal=(2*(double(x)+.5)/100-1)*tangent;
                const double vertical=(1-2*(double(y)+.5)/100)*tangent;
                const D3 ray=smoothglassfixture::unit({std::cos(angle)*horizontal-std::sin(angle),
                    vertical,-std::sin(angle)*horizontal-std::cos(angle)});
                const auto first=smoothglassfixture::triangleHit(*snapshot,camera,ray);
                if (!first.hit || identity(tiledObjects,pixel)!=first.object)
                { error = "GL rough tile edge changed full-frame raster visibility"; return false; }
                if (first.object!=snapshot->objects[0].id.value) continue;
                const auto expected=smoothglassfixture::traceRough(*snapshot,camera,ray,1.5,test.dispersion,test.roughness);
                if (!expected.optics.complete) { error="GL rough tile edge oracle did not complete";return false; }
                for (std::size_t c=0;c<3;++c) {
                    std::uint16_t value=0;std::memcpy(&value,tiledColor.bytes.data()+pixel*8+c*2,2);
                    if (std::abs(hdrimage::halfToFloat(value)-expected.optics.rgb[c])>.0015)
                    { error="GL rough tile edge Color differs from the independent full-frame ray oracle";return false; }
                }
            }
            const auto replay=backend.render(snapshot,nativePreparation.resources,100,100,inputs);
            arbitgpu::NativeRawPassPixels replayColor;
            std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> replaySchedule;
            arbitgpu::roughglasstiles::Aggregate replayAggregate;
            if (!replay.rendered || !replay.frame
                || !replay.frame->readRawPass(renderpassoutput::Output::Color,replayColor,error)
                || !arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*replay.frame,replaySchedule,replayAggregate,error)
                || replayColor.bytes!=tiledColor.bytes || replayAggregate.terminal!=aggregate.terminal
                || replayAggregate.masked!=aggregate.masked || replayAggregate.tail!=aggregate.tail
                || replayAggregate.receiptClosestHitQueries!=aggregate.receiptClosestHitQueries)
            { error="GL rough tiled native frame did not replay exact GPU Color and receipt totals";return false; }
            auto* context=glfwGetCurrentContext();
            glfwMakeContextCurrent(nullptr);
            const bool staleGpuAccepted=arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*tiled.frame,replaySchedule,replayAggregate,error);
            glfwMakeContextCurrent(context);
            if (staleGpuAccepted || replaySchedule || replayAggregate.completedTiles)
            { error="GL rough tile summary accepted an absent current GPU owner";return false; }
            error.clear();
            const auto stale = backend.render(std::make_shared<const Visual3DScene>(*snapshot),
                nativePreparation.resources,extent,extent,inputs);
            if (stale.rendered || stale.frame)
            { error = "GL rough stale prepared scene failure published a frame"; return false; }
        }
        nativeColors[row] = color.bytes;
        std::cout << "rough-glass-pixels backend=opengl row=" << row++ << " covered=" << coveredCount
            << " residual=" << stats.roughGlassMaximumEnergyResidual
            << " unresolved=" << stats.roughGlassMaximumUnresolvedPower << '\n';
    }
    if (!observedNonzeroTail || !observedOcclusion || nativeColors[0] == nativeColors[1]
        || nativeColors[1] == nativeColors[2] || nativeColors[1] == nativeColors[3]
        || nativeColors[1] == nativeColors[4])
    { error = "GL rough native footprint/dispersion/geometry/blocker controls or finite nonzero tail were not exercised"; return false; }
    {
        auto authored = smoothglassfixture::scene(.25f,1,20);
        smoothglassfixture::receiver(authored,-6,{1,1,1});
        auto snapshot = std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material = binding(snapshot,0,1.5f,.004f,error);
        if (!material) return false;
        const auto prepared = backend.prepare(snapshot,material->nativeProgram());
        if (!prepared.prepared) { error = prepared.error; return false; }
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
        inputs.linearColor = true;
        const auto smooth = backend.render(snapshot,prepared.resources,extent,extent,inputs);
        if (!smooth.rendered || !smooth.frame || smooth.stats.roughGlassReceiptComplete
            || smooth.stats.roughGlassReceiptBytes != 0 || smooth.stats.textureBytes != 0)
        { error = "GL roughness zero allocated or certified rough scratch instead of the released smooth path"; return false; }
        std::array<arbitgpu::NativeRawPassPixels,3> receipt;
        arbitgpu::NativeRawPassPixels objects, materials;
        if (arbitgpu::readOpenGlRoughGlassReceiptForFixture(*smooth.frame,receipt,objects,materials,error))
        { error = "GL diagnostic invented a rough receipt for exact zero roughness"; return false; }
    }
    // Full-size publication fixture with sparse authored glass. It exercises
    // every native job and full-size GPU storage without pretending to measure
    // the latency or convergence of a fully covered rough-glass canvas.
    {
        auto authored=smoothglassfixture::scene(.25f,1,20);
        smoothglassfixture::receiver(authored,-6,{1,1,1});
        auto& camera=authored.cameras[0];
        camera.transform.translation.x*=300;camera.transform.translation.z*=300;
        camera.verticalFovRadians=1;camera.nearPlane=10;camera.farPlane=4096;
        auto snapshot=std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material=binding(snapshot,.04f,1.5f,.004f,error);
        if (!material) return false;
        const auto prepared=backend.prepare(snapshot,material->nativeProgram());
        if (!prepared.prepared) {error=prepared.error;return false;}
        arbitgpu::NativeFixtureSceneRuntimeInputs inputs;inputs.linearColor=true;
        for (const auto extent : {std::array<std::uint32_t,2>{1920,1080},std::array<std::uint32_t,2>{3840,2160}}) {
            const auto complete=backend.render(snapshot,prepared.resources,extent[0],extent[1],inputs);
            if (!complete.rendered || !complete.frame || !complete.stats.roughGlassReceiptComplete
                || complete.frame->width()!=extent[0] || complete.frame->height()!=extent[1])
            {error="GL sparse full-size rough tile frame was not completed: "+complete.error;return false;}
            std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> schedule;
            arbitgpu::roughglasstiles::Aggregate aggregate;
            if (!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*complete.frame,schedule,aggregate,error)
                || aggregate.completedPixels!=std::uint64_t(extent[0])*extent[1]
                || aggregate.completedTiles!=schedule->tiles().size() || !aggregate.coveredPixels
                || aggregate.peakReceiptBytes>9708*48 || aggregate.reservedQueries!=schedule->reservedQueries())
            {error="GL full-size rough GPU frame has incomplete tile ownership/coverage";return false;}
            const auto descriptor=complete.frame->colorTextureDescriptor();
            int previousTexture=0,actualWidth=0,actualHeight=0;
            glGetIntegerv(GL_TEXTURE_BINDING_2D,&previousTexture);
            glBindTexture(GL_TEXTURE_2D,static_cast<unsigned>(descriptor.textureViewHandle));
            glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_WIDTH,&actualWidth);
            glGetTexLevelParameteriv(GL_TEXTURE_2D,0,GL_TEXTURE_HEIGHT,&actualHeight);
            glBindTexture(GL_TEXTURE_2D,static_cast<unsigned>(previousTexture));
            if (actualWidth!=int(extent[0]) || actualHeight!=int(extent[1]) || glGetError()!=GL_NO_ERROR)
            {error="GL rough completed descriptor does not own the actual full-size GPU image";return false;}
            std::cout<<"rough-glass-tiles backend=opengl extent="<<extent[0]<<'x'<<extent[1]
                <<" jobs="<<aggregate.completedTiles<<" covered="<<aggregate.coveredPixels
                <<" receiptClosestHitQueries="<<aggregate.receiptClosestHitQueries
                <<" reservedQueries="<<aggregate.reservedQueries<<" peakReceiptBytes="<<aggregate.peakReceiptBytes<<'\n';
        }
    }
    // Positive roughness remains bounded single scattering. Any failing pixel
    // refuses every public frame output, including an AOV inspection request.
    for (const bool internal : {false,true})
    {
        auto authored = smoothglassfixture::scene(.25f,1,20,internal);
        smoothglassfixture::receiver(authored,-6,{1,1,1});
        auto snapshot = std::make_shared<const Visual3DScene>(std::move(authored));
        const auto material = binding(snapshot,internal?.08f:.9f,1.5f,.004f,error);
        if (!material) return false;
        const auto prepared = backend.prepare(snapshot,material->nativeProgram());
        if (!prepared.prepared) { error = prepared.error; return false; }
        for (const auto output : {renderpassoutput::Output::Color,renderpassoutput::Output::Normal})
        {
            arbitgpu::NativeFixtureSceneRuntimeInputs inputs;
            inputs.linearColor = output == renderpassoutput::Output::Color;
            inputs.imageOutput = output;
            for (const auto failedExtent : {std::array<std::uint32_t,2>{extent,extent},std::array<std::uint32_t,2>{100,100}}) {
                const auto refused = backend.render(snapshot,prepared.resources,failedExtent[0],failedExtent[1],inputs);
                if (refused.rendered || refused.frame || refused.error.empty())
                { error = "GL rough failed tile/internal/unresolved transport published Color or an AOV"; return false; }
            }
        }
    }
    const auto walk=[](auto incoming,auto normal,double from,double to,double roughness,std::uint32_t seed) {
        return glasscaustictests::microfacetReference(incoming,normal,from,to,roughness,seed,32,16,.0001/128);
    };
    const auto read=[](const auto& frame,auto& receipts,auto& objects,auto& materials,std::string& error) {
        if(!arbitgpu::readOpenGlRoughGlassReceiptForFixture(*frame,receipts,objects,materials,error))return false;
        std::shared_ptr<const arbitgpu::roughglasstiles::Schedule> schedule;
        arbitgpu::roughglasstiles::Aggregate actual;
        if(!arbitgpu::readOpenGlRoughGlassTileSummaryForFixture(*frame,schedule,actual,error))return false;
        std::uint64_t glass=0,events=0,queries=0;
        for(std::size_t pixel=0;pixel<objects.bytes.size()/4;++pixel)if(identity(objects,pixel)==schedule->prepared().object.value) {
            ++glass;queries+=std::uint64_t(scalar(receipts[1],pixel,3));events+=std::uint64_t(scalar(receipts[2],pixel,3));
        }
        if(actual.selectionQueries!=schedule->pixels() || actual.transportStarts!=glass
            || actual.coveredPixels!=glass || actual.microEvents!=events || actual.receiptClosestHitQueries!=queries
            || actual.microWork!=events*arbitgpu::smoothglass::kCameraMicroWorkPerEvent)
        {error="GL v2 actual native walk starts or executed work differ from shared Color/receipt results";return false;}
        return true;
    };
    if(!smoothglassfixture::roughMultipleNativeCases(backend,walk,read,error))return false;
    error.clear();
    return true;
}
}
