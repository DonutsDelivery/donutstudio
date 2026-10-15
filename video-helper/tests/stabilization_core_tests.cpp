#include "stabilization.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
namespace stab = videohelper::stabilization;
constexpr unsigned width = 320, height = 240;
int failures = 0;
void check (bool condition, const char* message)
{
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
struct Pose { double x = 0, y = 0, angle = 0; };

// An analytic scene and image registration oracle authored independently of
// the feature tracker. Neither uses estimated transforms, feature locations,
// confidence or the core's rigid-fit implementation to measure image motion.
double scene (double x, double y, unsigned seed = 19)
{
    double result = 128;
    uint32_t random = seed;
    for (unsigned i = 0; i < 20; ++i)
    {
        random = random*1103515245+12345;
        const double fx = 0.08+double((random >> 8)%600)/1000;
        random = random*1103515245+12345;
        const double fy = -0.32+double((random >> 8)%640)/1000;
        random = random*1103515245+12345;
        result += (4+double(i%4))*std::sin(fx*x+fy*y+double(random%628)/100);
    }
    return std::clamp(result,0.0,255.0);
}
std::vector<uint8_t> makeFrame (Pose p, unsigned frame, bool foreground = true, unsigned seed = 19)
{
    std::vector<uint8_t> result(width*height);
    const double c = std::cos(p.angle), s = std::sin(p.angle);
    for (unsigned y = 0; y < height; ++y)
        for (unsigned x = 0; x < width; ++x)
        {
            const double px = x-(width-1)*0.5-p.x, py = y-(height-1)*0.5-p.y;
            const double wx = c*px+s*py, wy = -s*px+c*py;
            double value = scene(wx,wy,seed);
            const double foregroundX = 25*std::sin(frame*0.49);
            if (foreground && std::abs(x-(width-1)*0.5-foregroundX) < 42
                && std::abs(y-(height-1)*0.5) < 35)
                value = scene((x+frame*9)*1.1,(y-frame*5)*0.9,71);
            result[y*width+x] = uint8_t(std::lround(value));
        }
    return result;
}
stab::SourceIdentity source()
{
    stab::SourceIdentity s;
    // This synthetic frame source uses a fixed nonzero fixture content ID.
    s.sha256.fill(0x3b); s.streamIndex = 2; s.width = width; s.height = height;
    s.timeBaseNumerator = 1; s.timeBaseDenominator = 1000;
    s.inSeconds = 2; s.outSeconds = 20;
    return s;
}
stab::FrameKey frameKey (unsigned i) { return {i,2000+int64_t(i)*37+int64_t(i/3)*7}; }
stab::GrayFrame gray (const std::vector<uint8_t>& pixels)
{
    return {pixels.data(),width,height,width,pixels.size()};
}
double oracleError (const std::vector<uint8_t>& pixels, Pose p, double zoom)
{
    const double c = std::cos(p.angle), s = std::sin(p.angle);
    double error = 0; unsigned count = 0;
    for (unsigned y = 20; y+20 < height; y += 13)
        for (unsigned x = 20; x+20 < width; x += 13)
        {
            const double px = (x-(width-1)*0.5)/zoom-p.x;
            const double py = (y-(height-1)*0.5)/zoom-p.y;
            // Foreground never reaches these outer-background samples.
            if (std::abs(x-(width-1)*0.5) < 94 && std::abs(y-(height-1)*0.5) < 78) continue;
            const double delta = scene(c*px+s*py,-s*px+c*py)-pixels[y*width+x];
            error += delta*delta; ++count;
        }
    return error/count;
}
Pose registerBackground (const std::vector<uint8_t>& pixels, double zoom)
{
    Pose best; double cost = std::numeric_limits<double>::infinity();
    // Full-image photometric search, distinct from patch correspondence/RANSAC.
    for (int angle = -5; angle <= 5; ++angle)
        for (int y = -5; y <= 5; ++y)
            for (int x = -5; x <= 5; ++x)
            {
                const Pose candidate {x*2.0,y*2.0,angle*0.008};
                const double e = oracleError(pixels,candidate,zoom);
                if (e < cost) { best = candidate; cost = e; }
            }
    double xyStep = 1, angleStep = 0.004;
    for (unsigned level = 0; level < 6; ++level)
    {
        for (unsigned repeat = 0; repeat < 4; ++repeat)
        {
            const auto center = best;
            for (int angle = -1; angle <= 1; ++angle)
                for (int y = -1; y <= 1; ++y)
                    for (int x = -1; x <= 1; ++x)
                    {
                        const Pose candidate {center.x+x*xyStep,center.y+y*xyStep,center.angle+angle*angleStep};
                        const double e = oracleError(pixels,candidate,zoom);
                        if (e < cost) { best = candidate; cost = e; }
                    }
        }
        xyStep *= 0.5; angleStep *= 0.5;
    }
    return best;
}
double trajectoryRms (const std::vector<Pose>& poses)
{
    Pose mean;
    for (auto p : poses) { mean.x += p.x; mean.y += p.y; mean.angle += p.angle; }
    mean.x /= poses.size(); mean.y /= poses.size(); mean.angle /= poses.size();
    double sum = 0;
    for (auto p : poses)
        sum += (p.x-mean.x)*(p.x-mean.x)+(p.y-mean.y)*(p.y-mean.y)
             + 10000*(p.angle-mean.angle)*(p.angle-mean.angle);
    return std::sqrt(sum/poses.size());
}
void motionOracle()
{
    constexpr unsigned count = 48;
    stab::Analyzer analyzer(source());
    std::vector<std::vector<uint8_t>> frames;
    std::vector<Pose> before, after;
    std::string error;
    for (unsigned i = 0; i < count; ++i)
    {
        const Pose camera {4.8*std::sin(i*2.17)+1.2*std::cos(i*0.43),
            3.6*std::cos(i*1.63),0.014*std::sin(i*1.79)+0.006*(i%2 ? 1 : -1)};
        frames.push_back(makeFrame(camera,i));
        check(analyzer.append(gray(frames.back()),frameKey(i),error),error.c_str());
        const auto measured = registerBackground(frames.back(),1);
        check(std::hypot(measured.x-camera.x,measured.y-camera.y) < 0.4
              && std::abs(measured.angle-camera.angle) < 0.004,"Independent raw-image registration resolves fixture camera");
        before.push_back(measured);
    }
    stab::Track track;
    if (!analyzer.finish(track,error)) { check(false,error.c_str()); return; }
    unsigned accepted = 0;
    for (const auto& f : track.frames) if (f.status == stab::FrameStatus::tracked) ++accepted;
    check(accepted >= count*8/10,"Textured background survives a moving foreground");
    stab::RenderPlan plan;
    if (!stab::makeRenderPlan(track,0.35,plan,error)) { check(false,error.c_str()); return; }
    check(plan.smoothingRadius == 21 && plan.zoom <= 1.2,"Strength radius and clip-wide crop remain bounded");
    for (unsigned i = 0; i < count; ++i)
    {
        std::vector<uint8_t> output(width*height);
        const stab::ConstPlane input {frames[i].data(),width,height,1,width,frames[i].size(),stab::SampleType::u8};
        const stab::MutablePlane destination {output.data(),width,height,1,width,output.size(),stab::SampleType::u8};
        if (!stab::warpPlane(plan,i,frameKey(i),input,destination,error)) { check(false,error.c_str()); return; }
        after.push_back(registerBackground(output,plan.zoom));
    }
    const double raw = trajectoryRms(before), stabilized = trajectoryRms(after);
    std::cout << "background trajectory RMS before=" << raw << " after=" << stabilized << " zoom=" << plan.zoom << '\n';
    check(stabilized < raw*0.45,"Independent image oracle detects reduced background XY/rotation RMS");
}
void validationAndResets()
{
    std::string error;
    auto decimalSource = source();
    decimalSource.timeBaseDenominator = 10; decimalSource.inSeconds = 0.1;
    stab::Analyzer decimalAnalyzer(decimalSource);
    auto decimalFrame = makeFrame({},0,false);
    check(decimalAnalyzer.append(gray(decimalFrame),{0,1},error),"Decimal trim boundary agrees with decoder double seconds");
    stab::Analyzer analyzer(source());
    auto a = makeFrame({},0,false), b = makeFrame({2,-1,0.005},1,false);
    check(!analyzer.append(gray(a),{1,2000},error),"First presentation ordinal must be zero");
    check(!analyzer.append(gray(a),{0,1999},error),"Pre-trim frame is rejected");
    check(analyzer.append(gray(a),frameKey(0),error),"Anchor append");
    check(!analyzer.append(gray(b),{1,1999},error),"Reordered timestamp is rejected");
    check(!analyzer.append(gray(b),frameKey(1),error,[] { return true; }),"Cancellation rejects append");
    check(analyzer.append(gray(b),frameKey(1),error),"Cancellation leaves no admitted partial frame");
    std::vector<uint8_t> flat(width*height,0);
    check(analyzer.append(gray(flat),frameKey(2),error),"Cut resets rather than inventing motion");
    check(analyzer.append(gray(flat),frameKey(3),error),"Low texture resets safely");
    check(analyzer.append(gray(a),frameKey(4),error),"Recovery from full occlusion resets safely");
    auto covered = makeFrame({},5,false,71);
    check(analyzer.append(gray(covered),frameKey(5),error),"Textured full occlusion admits a safe reset");
    check(analyzer.append(gray(a),frameKey(6),error),"Background return admits a safe reset");
    stab::Track track;
    if (!analyzer.finish(track,error)) { check(false,error.c_str()); return; }
    check(track.frames[5].status != stab::FrameStatus::tracked
          && track.frames[6].status != stab::FrameStatus::tracked,"Unrelated textured full occlusion does not acquire background motion");
    check(track.frames[2].status == stab::FrameStatus::cut,"Histogram cut marked explicitly");
    check(track.frames[3].status == stab::FrameStatus::uncertain,"Low texture marked explicitly");
    for (size_t i = 2; i < 5; ++i)
        check(track.frames[i].camera.x == 0 && track.frames[i].camera.y == 0
              && track.frames[i].camera.angle == 0,"Cut/occlusion cannot teleport the camera path");
    std::vector<uint8_t> encoded;
    check(stab::encodeTrack(track,encoded,error),"Track encoding");
    stab::Track decoded;
    check(stab::decodeTrack(encoded.data(),encoded.size(),decoded,error),"Track decoding");
    check(stab::sameSource(track.source,decoded.source) && decoded.frames.size() == track.frames.size(),"Exact identity/extent/SAR/timebase/range survives round trip");
    auto changed = decoded.source; changed.sha256[0] ^= 1;
    check(!stab::sameSource(track.source,changed),"Changed source bytes cannot hit analysis cache");
    changed = decoded.source; changed.inSeconds = std::nextafter(changed.inSeconds,3.0);
    check(!stab::sameSource(track.source,changed),"Trim identity retains full precision");
    check(!stab::matchesFrame(decoded,1,{1,decoded.frames[1].key.pts+1},error),"Render PTS mismatch is rejected");
    check(!stab::matchesFrame(decoded,1,{2,decoded.frames[1].key.pts},error),"Render ordinal mismatch is rejected");
    auto broken = encoded; broken[55] ^= 1;
    check(!stab::decodeTrack(broken.data(),broken.size(),decoded,error),"Corrupt track checksum is rejected");
    broken = encoded; broken[4] = 99;
    check(!stab::decodeTrack(broken.data(),broken.size(),decoded,error),"Unsupported schema is rejected");
    broken = encoded; broken.push_back(0);
    check(!stab::decodeTrack(broken.data(),broken.size(),decoded,error),"Trailing bytes are rejected");
    broken = encoded; broken[12] = broken[13] = broken[14] = broken[15] = 255;
    check(!stab::decodeTrack(broken.data(),broken.size(),decoded,error),"Oversized record count is rejected before allocation");
    auto bad = track; bad.frames[1].camera.angle = std::numeric_limits<double>::quiet_NaN();
    check(!stab::validateTrack(bad,error),"Non-finite camera fit is rejected");
    bad = track; bad.frames[1].key.ordinal = 2;
    check(!stab::validateTrack(bad,error),"Corrupt presentation order is rejected");
    stab::RenderPlan plan;
    check(stab::makeRenderPlan(track,0,plan,error) && plan.smoothingRadius == 1,"Zero strength retains released minimum radius");
    check(stab::makeRenderPlan(track,1,plan,error) && plan.smoothingRadius == 60,"Full strength retains released maximum radius");
    check(!stab::makeRenderPlan(track,0.5,plan,error,1.35,[] { return true; }),"Cancellation rejects smoothing");
    bad = track; bad.frames[1].camera.x = 1000;
    check(!stab::makeRenderPlan(bad,1,plan,error),"Excessive border crop fails visibly");
}
void sampleRepresentations()
{
    std::array<uint8_t,12> padded {1,2,3,4,90,91,5,6,7,8,92,93};
    std::array<uint8_t,32> digestA, digestB;
    std::string digestError;
    const stab::GrayFrame grayPadded {padded.data(),4,2,6,padded.size()};
    check(stab::grayFrameDigest(grayPadded,digestA,digestError),"Visible-row digest admission");
    padded[4] ^= 1; padded[11] ^= 1;
    check(stab::grayFrameDigest(grayPadded,digestB,digestError) && digestA == digestB,"Stride padding never affects replay digest");
    padded[6] ^= 1;
    check(stab::grayFrameDigest(grayPadded,digestB,digestError) && digestA != digestB,"Visible-row mutation affects replay digest");
    auto s = source(); s.sarNumerator = 4; s.sarDenominator = 3;
    stab::Track track; track.source = s;
    stab::TrackFrame anchor; anchor.key = frameKey(0); anchor.graySha256.fill(0x2f); track.frames.push_back(anchor);
    stab::RenderPlan plan; std::string error;
    check(stab::makeRenderPlan(track,0.4,plan,error),"Non-square pixel plan admission");
    std::vector<uint16_t> input(width*height), output(width*height);
    for (size_t i = 0; i < input.size(); ++i) input[i] = uint16_t(i%1024);
    const stab::ConstPlane u16 {input.data(),width,height,1,width*2,input.size()*2,stab::SampleType::u16};
    const stab::MutablePlane u16out {output.data(),width,height,1,width*2,output.size()*2,stab::SampleType::u16};
    check(stab::warpPlane(plan,0,frameKey(0),u16,u16out,error) && input == output,"Identity warp retains 10-bit integer samples");
    std::vector<float> floating(width*height,4.5f), rendered(width*height);
    const stab::ConstPlane f32 {floating.data(),width,height,1,width*4,floating.size()*4,stab::SampleType::f32};
    const stab::MutablePlane f32out {rendered.data(),width,height,1,width*4,rendered.size()*4,stab::SampleType::f32};
    check(stab::warpPlane(plan,0,frameKey(0),f32,f32out,error) && floating == rendered,"Float HDR samples stay above one without clipping");
    floating[0] = std::numeric_limits<float>::infinity();
    check(!stab::warpPlane(plan,0,frameKey(0),f32,f32out,error),"Non-finite source samples fail");
    check(!stab::warpPlane(plan,0,frameKey(0),u16,u16out,error,[] { return true; }),"Cancellation rejects warp");
    const stab::MutablePlane overlapping {input.data(),width,height,1,width*2,input.size()*2,stab::SampleType::u16};
    check(!stab::warpPlane(plan,0,frameKey(0),u16,overlapping,error),"In-place warp is rejected");
}
}
int main()
{
    motionOracle(); validationAndResets(); sampleRepresentations();
    return failures ? 1 : 0;
}
