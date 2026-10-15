#include "stabilization.h"
#include "sha256.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace videohelper::stabilization
{
namespace
{
constexpr uint32_t maxExtent = 16384;
constexpr uint64_t maxPixels = 64 * 1024 * 1024;
constexpr unsigned analysisExtent = 640, featureLimit = 288;

bool fail (std::string& error, const char* message) { error = message; return false; }
bool stopped (const CancelCheck& c) { return c && c(); }
bool finite (const RigidTransform& t)
{
    return std::isfinite(t.x) && std::isfinite(t.y) && std::isfinite(t.angle);
}
bool validSource (const SourceIdentity& s, std::string& error)
{
    if (s.streamIndex < 0 || s.width < 2 || s.height < 2
        || s.width > maxExtent || s.height > maxExtent
        || uint64_t(s.width) * s.height > maxPixels)
        return fail(error, "Invalid stabilization stream or visible extent");
    if (s.sarNumerator <= 0 || s.sarDenominator <= 0
        || s.timeBaseNumerator <= 0 || s.timeBaseDenominator <= 0)
        return fail(error, "Invalid stabilization sample aspect ratio or time base");
    const auto sar = double(s.sarNumerator) / s.sarDenominator;
    if (sar < 0.1 || sar > 10 || !std::isfinite(s.inSeconds)
        || !std::isfinite(s.outSeconds) || s.inSeconds < 0
        || s.outSeconds <= s.inSeconds)
        return fail(error, "Invalid stabilization range or sample aspect ratio");
    if (std::all_of(s.sha256.begin(), s.sha256.end(), [](uint8_t b) { return b == 0; }))
        return fail(error, "Stabilization requires a SHA-256 content identity");
    return true;
}
bool validKey (const SourceIdentity& s, FrameKey k, std::string& error)
{
    if (k.pts == std::numeric_limits<int64_t>::min())
        return fail(error, "Stabilization frame has no presentation timestamp");
    // Match the decoder's double seconds comparison, including decimal trim
    // boundaries. Extra intermediate precision can reject a frame the decoder
    // included at an exactly matching double timestamp.
    const double sec = double(k.pts) * (double(s.timeBaseNumerator) / s.timeBaseDenominator);
    if (sec < s.inSeconds || sec >= s.outSeconds)
        return fail(error, "Stabilization frame falls outside the requested range");
    return true;
}
RigidTransform compose (const RigidTransform& a, const RigidTransform& b)
{
    const double c = std::cos(a.angle), s = std::sin(a.angle);
    return {c*b.x - s*b.y + a.x, s*b.x + c*b.y + a.y, a.angle + b.angle};
}
RigidTransform inverse (const RigidTransform& t)
{
    const double c = std::cos(t.angle), s = std::sin(t.angle);
    return {-c*t.x - s*t.y, s*t.x - c*t.y, -t.angle};
}
struct Point { double x = 0, y = 0; };
Point apply (const RigidTransform& t, Point p)
{
    const double c = std::cos(t.angle), s = std::sin(t.angle);
    return {c*p.x - s*p.y + t.x, s*p.x + c*p.y + t.y};
}
double distance2 (Point a, Point b)
{
    return (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y);
}
struct Image
{
    unsigned width = 0, height = 0;
    std::vector<float> pixels;
    float at (int x, int y) const { return pixels[size_t(y)*width + x]; }
    double sample (double x, double y) const
    {
        const int ix = int(x), iy = int(y);
        const auto dx = x-ix, dy = y-iy;
        return (1-dy)*((1-dx)*at(ix,iy) + dx*at(ix+1,iy))
             + dy*((1-dx)*at(ix,iy+1) + dx*at(ix+1,iy+1));
    }
    bool contains (Point p, double margin) const
    {
        return p.x >= margin && p.y >= margin
            && p.x < width-1-margin && p.y < height-1-margin;
    }
};
using Pyramid = std::vector<Image>;
bool makePyramid (const GrayFrame& frame, Pyramid& result, const CancelCheck& cancel)
{
    const double scale = std::min(1.0, double(analysisExtent) / std::max(frame.width, frame.height));
    Image image;
    image.width = std::max(1u, unsigned(std::lround(frame.width * scale)));
    image.height = std::max(1u, unsigned(std::lround(frame.height * scale)));
    image.pixels.resize(size_t(image.width)*image.height);
    // Area averages avoid aliasing fine texture into false coarse matches.
    for (unsigned y = 0; y < image.height; ++y)
    {
        if (stopped(cancel)) return false;
        const unsigned y0 = uint64_t(y)*frame.height/image.height;
        const unsigned y1 = std::max(y0+1, unsigned(uint64_t(y+1)*frame.height/image.height));
        for (unsigned x = 0; x < image.width; ++x)
        {
            const unsigned x0 = uint64_t(x)*frame.width/image.width;
            const unsigned x1 = std::max(x0+1, unsigned(uint64_t(x+1)*frame.width/image.width));
            double sum = 0;
            for (unsigned sy = y0; sy < y1; ++sy)
                for (unsigned sx = x0; sx < x1; ++sx)
                    sum += frame.pixels[size_t(sy)*frame.strideBytes + sx];
            image.pixels[size_t(y)*image.width+x] = float(sum/((y1-y0)*(x1-x0)));
        }
    }
    result.push_back(std::move(image));
    while (result.size() < 4 && result.back().width >= 96 && result.back().height >= 96)
    {
        const auto& prev = result.back();
        Image next;
        next.width = prev.width/2; next.height = prev.height/2;
        next.pixels.resize(size_t(next.width)*next.height);
        for (unsigned y = 0; y < next.height; ++y)
            for (unsigned x = 0; x < next.width; ++x)
                next.pixels[size_t(y)*next.width+x] = 0.25f*(prev.at(x*2,y*2)
                    + prev.at(x*2+1,y*2) + prev.at(x*2,y*2+1) + prev.at(x*2+1,y*2+1));
        result.push_back(std::move(next));
    }
    return !stopped(cancel);
}
std::vector<Point> selectFeatures (const Image& image)
{
    struct Candidate { Point p; double score; };
    std::array<std::vector<Candidate>, 48> cells;
    for (unsigned y = 7; y+7 < image.height; y += 3)
        for (unsigned x = 7; x+7 < image.width; x += 3)
        {
            double xx = 0, xy = 0, yy = 0;
            for (int oy = -2; oy <= 2; ++oy)
                for (int ox = -2; ox <= 2; ++ox)
                {
                    const int px = int(x)+ox, py = int(y)+oy;
                    const double gx = 0.5*(image.at(px+1,py)-image.at(px-1,py));
                    const double gy = 0.5*(image.at(px,py+1)-image.at(px,py-1));
                    xx += gx*gx; xy += gx*gy; yy += gy*gy;
                }
            const double score = 0.5*(xx+yy-std::sqrt((xx-yy)*(xx-yy)+4*xy*xy))/25;
            if (score > 25)
                cells[std::min(5u,y*6/image.height)*8 + std::min(7u,x*8/image.width)]
                    .push_back({{double(x),double(y)},score});
        }
    std::vector<Point> result;
    for (auto& cell : cells)
    {
        std::sort(cell.begin(), cell.end(), [](const Candidate& a, const Candidate& b)
                  { return a.score > b.score; });
        unsigned kept = 0;
        for (const auto& c : cell)
        {
            if (std::none_of(result.begin(), result.end(), [&](Point p) { return distance2(p,c.p) < 64; }))
            {
                result.push_back(c.p);
                if (++kept == featureLimit/cells.size()) break;
            }
        }
    }
    return result;
}
// Mean-subtracted patch error permits a global exposure offset. Subpixel
// refinement uses the actual image gradient, not an interpolated integer flow.
double patchError (const Image& a, const Image& b, Point p, Point q)
{
    double sum = 0, sum2 = 0;
    for (int y = -3; y <= 3; ++y)
        for (int x = -3; x <= 3; ++x)
        {
            const double d = a.sample(p.x+x,p.y+y)-b.sample(q.x+x,q.y+y);
            sum += d; sum2 += d*d;
        }
    return std::max(0.0, (sum2-sum*sum/49)/49);
}
bool trackPoint (const Pyramid& a, const Pyramid& b, Point p, Point& q)
{
    int level = int(std::min(a.size(), b.size()))-1;
    while (level > 0 && !a[size_t(level)].contains({(p.x+0.5)/std::pow(2,level)-0.5,(p.y+0.5)/std::pow(2,level)-0.5},5)) --level;
    Point estimate {(p.x+0.5)/std::pow(2,level)-0.5,(p.y+0.5)/std::pow(2,level)-0.5};
    for (int l = level; l >= 0; --l)
    {
        const auto& src = a[size_t(l)]; const auto& dst = b[size_t(l)];
        Point origin {(p.x+0.5)/std::pow(2,l)-0.5,(p.y+0.5)/std::pow(2,l)-0.5};
        if (!src.contains(origin,4)) return false;
        if (l != level) { estimate.x = (estimate.x+0.5)*2-0.5; estimate.y = (estimate.y+0.5)*2-0.5; }
        const int radius = l == level ? 6 : 2;
        Point best {}; double cost = std::numeric_limits<double>::infinity();
        for (int y = -radius; y <= radius; ++y)
            for (int x = -radius; x <= radius; ++x)
            {
                Point candidate {estimate.x+x,estimate.y+y};
                if (!dst.contains(candidate,4)) continue;
                const double e = patchError(src,dst,origin,candidate);
                if (e < cost) { cost = e; best = candidate; }
            }
        if (!std::isfinite(cost)) return false;
        estimate = best;
        for (unsigned iteration = 0; iteration < 8; ++iteration)
        {
            if (!dst.contains(estimate,5)) return false;
            double bias = 0;
            for (int y = -3; y <= 3; ++y)
                for (int x = -3; x <= 3; ++x)
                    bias += src.sample(origin.x+x,origin.y+y)-dst.sample(estimate.x+x,estimate.y+y);
            bias /= 49;
            double xx = 0, xy = 0, yy = 0, ex = 0, ey = 0;
            for (int y = -3; y <= 3; ++y)
                for (int x = -3; x <= 3; ++x)
                {
                    const double px = estimate.x+x, py = estimate.y+y;
                    const double gx = 0.5*(dst.sample(px+1,py)-dst.sample(px-1,py));
                    const double gy = 0.5*(dst.sample(px,py+1)-dst.sample(px,py-1));
                    const double e = src.sample(origin.x+x,origin.y+y)-dst.sample(px,py)-bias;
                    xx += gx*gx; xy += gx*gy; yy += gy*gy; ex += e*gx; ey += e*gy;
                }
            const double det = xx*yy-xy*xy;
            if (det < 1e-6) return false;
            double dx = (yy*ex-xy*ey)/det, dy = (xx*ey-xy*ex)/det;
            const double length = std::hypot(dx,dy);
            if (!std::isfinite(length)) return false;
            if (length > 1) { dx /= length; dy /= length; }
            Point refined {estimate.x+dx,estimate.y+dy};
            if (!dst.contains(refined,5)) return false;
            if (patchError(src,dst,origin,refined) > patchError(src,dst,origin,estimate)+0.01) break;
            estimate = refined;
            if (length < 0.02) break;
        }
    }
    if (patchError(a[0],b[0],p,estimate) > 500) return false;
    q = estimate;
    return true;
}
struct Match { Point from, to; unsigned cell = 0; };
RigidTransform fit (const std::vector<Match>& matches, const std::vector<size_t>& indices)
{
    Point a {}, b {};
    for (auto i : indices) { a.x += matches[i].from.x; a.y += matches[i].from.y;
                            b.x += matches[i].to.x; b.y += matches[i].to.y; }
    a.x /= indices.size(); a.y /= indices.size(); b.x /= indices.size(); b.y /= indices.size();
    double dot = 0, cross = 0;
    for (auto i : indices)
    {
        const Point p {matches[i].from.x-a.x,matches[i].from.y-a.y};
        const Point q {matches[i].to.x-b.x,matches[i].to.y-b.y};
        dot += p.x*q.x+p.y*q.y; cross += p.x*q.y-p.y*q.x;
    }
    const double angle = std::atan2(cross,dot), c = std::cos(angle), s = std::sin(angle);
    return {b.x-c*a.x+s*a.y,b.y-s*a.x-c*a.y,angle};
}
std::vector<size_t> inliers (const std::vector<Match>& m, const RigidTransform& t, double threshold)
{
    std::vector<size_t> result;
    for (size_t i = 0; i < m.size(); ++i)
        if (distance2(apply(t,m[i].from),m[i].to) <= threshold*threshold) result.push_back(i);
    return result;
}
bool estimate (const Pyramid& a, const Pyramid& b, const SourceIdentity& source,
               RigidTransform& motion, double& confidence, const CancelCheck& cancel)
{
    const auto features = selectFeatures(a[0]);
    const double sx = double(source.width)/a[0].width;
    const double sy = double(source.height)/a[0].height;
    const double sar = double(source.sarNumerator)/source.sarDenominator;
    const double threshold = 2.0*std::max(sx*sar,sy);
    std::vector<Match> matches;
    for (auto p : features)
    {
        if (stopped(cancel)) return false;
        Point q, back;
        if (!trackPoint(a,b,p,q) || !trackPoint(b,a,q,back) || distance2(p,back) > 1.5*1.5) continue;
        const auto centered = [&](Point v) -> Point
        {
            return {((v.x+0.5)*sx-source.width*0.5)*sar,
                    (v.y+0.5)*sy-source.height*0.5};
        };
        matches.push_back({centered(p),centered(q),
            std::min(2u,unsigned(p.y)*3/a[0].height)*4+std::min(3u,unsigned(p.x)*4/a[0].width)});
    }
    if (matches.size() < 12) return false;
    std::vector<size_t> best;
    unsigned bestScore = 0; uint32_t random = 0x91a772b3;
    for (unsigned trial = 0; trial < 512; ++trial)
    {
        random = random*1664525u+1013904223u; const size_t i = random%matches.size();
        random = random*1664525u+1013904223u; const size_t j = random%matches.size();
        if (i == j || distance2(matches[i].from,matches[j].from) < threshold*threshold*64) continue;
        const auto candidate = fit(matches,{i,j});
        if (std::abs(candidate.angle) > 0.25) continue;
        auto accepted = inliers(matches,candidate,threshold);
        std::array<unsigned,12> bins {};
        for (auto k : accepted) ++bins[matches[k].cell];
        unsigned score = 0;
        for (auto n : bins) score += std::min(4u,n);
        if (score > bestScore || (score == bestScore && accepted.size() > best.size()))
        { best = std::move(accepted); bestScore = score; }
    }
    if (best.size() < 12) return false;
    for (unsigned iteration = 0; iteration < 3; ++iteration)
    {
        motion = fit(matches,best);
        best = inliers(matches,motion,threshold);
        if (best.size() < 12) return false;
    }
    motion = fit(matches,best);
    std::array<bool,12> cells {};
    double minX = 1e30, maxX = -1e30, minY = 1e30, maxY = -1e30, sum2 = 0;
    for (auto i : best)
    {
        const auto p = matches[i].from;
        cells[matches[i].cell] = true;
        minX = std::min(minX,p.x); maxX = std::max(maxX,p.x);
        minY = std::min(minY,p.y); maxY = std::max(maxY,p.y);
        sum2 += distance2(apply(motion,p),matches[i].to);
    }
    const double ratio = double(best.size())/matches.size();
    if (!finite(motion) || ratio < 0.45 || std::count(cells.begin(),cells.end(),true) < 5
        || maxX-minX < source.width*sar*0.35 || maxY-minY < source.height*0.35
        || std::abs(motion.x) > source.width*sar*0.2 || std::abs(motion.y) > source.height*0.2
        || std::abs(motion.angle) > 0.25)
        return false;
    confidence = ratio * std::exp(-sum2/(best.size()*threshold*threshold));
    return true;
}
double histogramDifference (const Image& a, const Image& b)
{
    std::array<double,16> ah {}, bh {};
    for (auto v : a.pixels) ++ah[std::min(15u,unsigned(v)/16)];
    for (auto v : b.pixels) ++bh[std::min(15u,unsigned(v)/16)];
    double result = 0;
    for (size_t i = 0; i < ah.size(); ++i)
        result += std::abs(ah[i]/a.pixels.size()-bh[i]/b.pixels.size());
    return result*0.5;
}
} // namespace

struct Analyzer::Impl
{
    Track track;
    Pyramid previous;
    std::string initializationError;
    explicit Impl (const SourceIdentity& s) { track.source = s; validSource(s,initializationError); }
};
Analyzer::Analyzer (const SourceIdentity& s) : impl(std::make_unique<Impl>(s)) {}
Analyzer::~Analyzer() = default;
Analyzer::Analyzer (Analyzer&&) noexcept = default;
Analyzer& Analyzer::operator= (Analyzer&&) noexcept = default;

bool Analyzer::append (const GrayFrame& f, FrameKey key, std::string& error, const CancelCheck& cancel)
{
    if (!impl) return fail(error,"Stabilization analyzer was moved");
    if (!impl->initializationError.empty()) { error = impl->initializationError; return false; }
    const auto& source = impl->track.source;
    if (!f.pixels || f.width != source.width || f.height != source.height
        || f.strideBytes < f.width || f.strideBytes > maximumTrackBytes
        || size_t(f.height-1) > (SIZE_MAX-f.width)/f.strideBytes
        || f.byteCount < size_t(f.height-1)*f.strideBytes+f.width)
        return fail(error,"Invalid stabilization grayscale frame storage");
    if (impl->track.frames.size() >= maximumFrames) return fail(error,"Stabilization frame limit exceeded");
    if (!validKey(source,key,error)) return false;
    if (key.ordinal != impl->track.frames.size()
        || (!impl->track.frames.empty() && key.pts < impl->track.frames.back().key.pts))
        return fail(error,"Stabilization presentation order mismatch");
    if (stopped(cancel)) return fail(error,"Stabilization cancelled");
    Pyramid current;
    if (!makePyramid(f,current,cancel)) return fail(error,"Stabilization cancelled");
    TrackFrame record; record.key = key;
    if (!grayFrameDigest(f,record.graySha256,error)) return false;
    if (!impl->previous.empty())
    {
        const auto& last = impl->track.frames.back();
        record.segment = last.segment;
        RigidTransform motion;
        const bool cut = histogramDifference(impl->previous[0],current[0]) > 0.55;
        const bool tracked = !cut && estimate(impl->previous,current,source,motion,record.confidence,cancel);
        if (stopped(cancel)) return fail(error,"Stabilization cancelled");
        if (tracked)
        {
            record.camera = compose(motion,last.camera);
            record.status = FrameStatus::tracked;
            if (!finite(record.camera)) return fail(error,"Non-finite stabilization camera path");
        }
        else
        {
            record.segment = last.segment+1;
            record.status = cut ? FrameStatus::cut : FrameStatus::uncertain;
            record.confidence = 0;
        }
    }
    if (stopped(cancel)) return fail(error,"Stabilization cancelled");
    impl->track.frames.push_back(record);
    impl->previous = std::move(current);
    error.clear();
    return true;
}
bool Analyzer::finish (Track& result, std::string& error) const
{
    if (!impl) return fail(error,"Stabilization analyzer was moved");
    if (!impl->initializationError.empty()) { error = impl->initializationError; return false; }
    if (!validateTrack(impl->track,error)) return false;
    result = impl->track;
    return true;
}
bool sameSource (const SourceIdentity& a, const SourceIdentity& b)
{
    return a.sha256 == b.sha256 && a.streamIndex == b.streamIndex
        && a.width == b.width && a.height == b.height
        && a.sarNumerator == b.sarNumerator && a.sarDenominator == b.sarDenominator
        && a.timeBaseNumerator == b.timeBaseNumerator && a.timeBaseDenominator == b.timeBaseDenominator
        && a.inSeconds == b.inSeconds && a.outSeconds == b.outSeconds;
}
bool validateTrack (const Track& t, std::string& error)
{
    if (t.schema != trackSchema || t.estimator != estimatorRevision)
        return fail(error,"Unsupported stabilization track version");
    if (!validSource(t.source,error)) return false;
    if (t.frames.empty() || t.frames.size() > maximumFrames)
        return fail(error,"Invalid stabilization track frame count");
    for (size_t i = 0; i < t.frames.size(); ++i)
    {
        const auto& f = t.frames[i];
        if (!validKey(t.source,f.key,error)) return false;
        if (f.key.ordinal != i || (i && f.key.pts < t.frames[i-1].key.pts))
            return fail(error,"Stabilization track presentation order mismatch");
        if (std::all_of(f.graySha256.begin(),f.graySha256.end(),[](uint8_t b) { return b == 0; }))
            return fail(error,"Stabilization frame is missing its Gray8 digest");
        if (!finite(f.camera) || std::abs(f.camera.x) > 1e9 || std::abs(f.camera.y) > 1e9
            || std::abs(f.camera.angle) > 1e6 || !std::isfinite(f.confidence)
            || f.confidence < 0 || f.confidence > 1
            || uint32_t(f.status) > uint32_t(FrameStatus::cut))
            return fail(error,"Invalid stabilization transform or confidence");
        const bool reset = f.status != FrameStatus::tracked;
        if ((!i && (f.segment != 0 || f.status != FrameStatus::anchor))
            || (i && (f.status == FrameStatus::anchor
                      || f.segment != t.frames[i-1].segment + unsigned(reset)))
            || (reset && (f.camera.x != 0 || f.camera.y != 0 || f.camera.angle != 0 || f.confidence != 0)))
            return fail(error,"Invalid stabilization segment boundary");
    }
    error.clear();
    return true;
}
bool matchesFrame (const Track& t, size_t index, FrameKey key, std::string& error)
{
    if (index >= t.frames.size() || t.frames[index].key.ordinal != key.ordinal
        || t.frames[index].key.pts != key.pts)
        return fail(error,"Stabilization render frame does not match analysis");
    error.clear(); return true;
}

namespace
{
constexpr size_t headerBytes = 92, frameBytes = 88, checksumBytes = 32;
void put32 (std::vector<uint8_t>& bytes, uint32_t value)
{
    for (unsigned i = 0; i < 4; ++i) bytes.push_back(uint8_t(value >> (8*i)));
}
void put64 (std::vector<uint8_t>& bytes, uint64_t value)
{
    for (unsigned i = 0; i < 8; ++i) bytes.push_back(uint8_t(value >> (8*i)));
}
void putDouble (std::vector<uint8_t>& bytes, double value)
{
    static_assert(sizeof(double) == 8 && std::numeric_limits<double>::is_iec559,
                  "Stabilization tracks require IEEE-754 binary64");
    uint64_t bits; std::memcpy(&bits,&value,8); put64(bytes,bits);
}
std::array<uint8_t,32> checksum (const void* data, size_t size)
{
    const auto hex = sha256Bytes(data,size);
    std::array<uint8_t,32> result {};
    const auto digit = [](char c) { return c <= '9' ? c-'0' : c-'a'+10; };
    for (size_t i = 0; i < result.size(); ++i) result[i] = uint8_t(digit(hex[i*2])*16+digit(hex[i*2+1]));
    return result;
}
struct Reader
{
    const uint8_t* data; size_t offset = 0;
    uint32_t u32()
    {
        uint32_t result = 0;
        for (unsigned i = 0; i < 4; ++i) result |= uint32_t(data[offset++]) << (8*i);
        return result;
    }
    int32_t i32() { const auto bits = u32(); int32_t result; std::memcpy(&result,&bits,4); return result; }
    uint64_t u64()
    {
        uint64_t result = 0;
        for (unsigned i = 0; i < 8; ++i) result |= uint64_t(data[offset++]) << (8*i);
        return result;
    }
    int64_t i64() { const auto bits = u64(); int64_t result; std::memcpy(&result,&bits,8); return result; }
    double real() { const auto bits = u64(); double result; std::memcpy(&result,&bits,8); return result; }
};
}
bool grayFrameDigest (const GrayFrame& f, std::array<uint8_t,32>& result, std::string& error)
{
    if (!f.pixels || !f.width || !f.height || f.width > maxExtent || f.height > maxExtent
        || uint64_t(f.width)*f.height > maxPixels || f.strideBytes < f.width
        || size_t(f.height-1) > (SIZE_MAX-f.width)/f.strideBytes
        || f.byteCount < size_t(f.height-1)*f.strideBytes+f.width)
        return fail(error,"Invalid stabilization grayscale digest storage");
    Sha256 hash;
    for (uint32_t y = 0; y < f.height; ++y)
        hash.update(f.pixels+size_t(y)*f.strideBytes,f.width);
    const auto hex = hash.finishHex();
    const auto digit = [](char c) { return c <= '9' ? c-'0' : c-'a'+10; };
    for (size_t i = 0; i < result.size(); ++i)
        result[i] = uint8_t(digit(hex[i*2])*16+digit(hex[i*2+1]));
    error.clear(); return true;
}
bool encodeTrack (const Track& track, std::vector<uint8_t>& result, std::string& error)
{
    if (!validateTrack(track,error)) return false;
    std::vector<uint8_t> bytes;
    bytes.reserve(headerBytes+frameBytes*track.frames.size()+checksumBytes);
    bytes.insert(bytes.end(),{'S','T','A','B'});
    put32(bytes,track.schema); put32(bytes,track.estimator); put32(bytes,uint32_t(track.frames.size()));
    const auto& s = track.source;
    bytes.insert(bytes.end(),s.sha256.begin(),s.sha256.end());
    put32(bytes,uint32_t(s.streamIndex)); put32(bytes,s.width); put32(bytes,s.height);
    put32(bytes,uint32_t(s.sarNumerator)); put32(bytes,uint32_t(s.sarDenominator));
    put32(bytes,uint32_t(s.timeBaseNumerator)); put32(bytes,uint32_t(s.timeBaseDenominator));
    putDouble(bytes,s.inSeconds); putDouble(bytes,s.outSeconds);
    for (const auto& f : track.frames)
    {
        put64(bytes,f.key.ordinal); put64(bytes,uint64_t(f.key.pts));
        putDouble(bytes,f.camera.x); putDouble(bytes,f.camera.y); putDouble(bytes,f.camera.angle);
        putDouble(bytes,f.confidence); put32(bytes,f.segment); put32(bytes,uint32_t(f.status));
        bytes.insert(bytes.end(),f.graySha256.begin(),f.graySha256.end());
    }
    const auto digest = checksum(bytes.data(),bytes.size());
    bytes.insert(bytes.end(),digest.begin(),digest.end());
    result = std::move(bytes); error.clear(); return true;
}
bool decodeTrack (const uint8_t* data, size_t size, Track& result, std::string& error)
{
    if (!data || size < headerBytes+checksumBytes || size > maximumTrackBytes
        || std::memcmp(data,"STAB",4) != 0)
        return fail(error,"Invalid stabilization track header or size");
    Reader r {data,4};
    Track track; track.schema = r.u32(); track.estimator = r.u32(); const auto count = r.u32();
    if (!count || count > maximumFrames || size != headerBytes+frameBytes*size_t(count)+checksumBytes)
        return fail(error,"Invalid stabilization track length");
    if (track.schema != trackSchema || track.estimator != estimatorRevision)
        return fail(error,"Unsupported stabilization track version");
    const auto digest = checksum(data,size-checksumBytes);
    if (!std::equal(digest.begin(),digest.end(),data+size-checksumBytes))
        return fail(error,"Stabilization track checksum mismatch");
    auto& s = track.source;
    std::copy(data+r.offset,data+r.offset+32,s.sha256.begin()); r.offset += 32;
    s.streamIndex = r.i32(); s.width = r.u32(); s.height = r.u32();
    s.sarNumerator = r.i32(); s.sarDenominator = r.i32();
    s.timeBaseNumerator = r.i32(); s.timeBaseDenominator = r.i32();
    s.inSeconds = r.real(); s.outSeconds = r.real();
    if (!validSource(s,error)) return false;
    track.frames.resize(count);
    for (auto& f : track.frames)
    {
        f.key.ordinal = r.u64(); f.key.pts = r.i64();
        f.camera.x = r.real(); f.camera.y = r.real(); f.camera.angle = r.real();
        f.confidence = r.real(); f.segment = r.u32(); f.status = FrameStatus(r.u32());
        std::copy(data+r.offset,data+r.offset+32,f.graySha256.begin()); r.offset += 32;
    }
    if (!validateTrack(track,error)) return false;
    result = std::move(track); error.clear(); return true;
}

bool makeRenderPlan (const Track& track, double strength, RenderPlan& result, std::string& error,
                     double maxZoom, const CancelCheck& cancel)
{
    if (!validateTrack(track,error)) return false;
    if (!std::isfinite(strength) || strength < 0 || strength > 1
        || !std::isfinite(maxZoom) || maxZoom < 1 || maxZoom > 2)
        return fail(error,"Invalid stabilization strength or crop limit");
    RenderPlan plan; plan.source = track.source;
    plan.smoothingRadius = unsigned(std::clamp(std::lround(strength*60),1l,60l));
    plan.keys.reserve(track.frames.size()); plan.outputToSource.reserve(track.frames.size());
    size_t begin = 0;
    while (begin < track.frames.size())
    {
        size_t end = begin+1;
        while (end < track.frames.size() && track.frames[end].segment == track.frames[begin].segment) ++end;
        // Prefix sums keep a 60-frame radius linear in the clip length.
        std::vector<RigidTransform> sums(end-begin+1);
        for (size_t i = begin; i < end; ++i)
        {
            const auto& path = track.frames[i].camera; const auto& prev = sums[i-begin];
            sums[i-begin+1] = {prev.x+path.x,prev.y+path.y,prev.angle+path.angle};
        }
        for (size_t i = begin; i < end; ++i)
        {
            if (stopped(cancel)) return fail(error,"Stabilization cancelled");
            const size_t left = i-begin > plan.smoothingRadius ? i-plan.smoothingRadius : begin;
            const size_t right = std::min(end,i+plan.smoothingRadius+1);
            const auto& a = sums[left-begin]; const auto& b = sums[right-begin];
            const double n = double(right-left);
            const RigidTransform smooth {(b.x-a.x)/n,(b.y-a.y)/n,(b.angle-a.angle)/n};
            // Source C, desired S: inverse mapping C * inverse(S).
            const auto mapping = compose(track.frames[i].camera,inverse(smooth));
            if (!finite(mapping)) return fail(error,"Non-finite stabilization render transform");
            plan.keys.push_back(track.frames[i].key); plan.outputToSource.push_back(mapping);
        }
        begin = end;
    }
    const double sar = double(plan.source.sarNumerator)/plan.source.sarDenominator;
    const double halfX = (plan.source.width-1)*sar*0.5, halfY = (plan.source.height-1)*0.5;
    // For q = R*p/zoom + t, extrema of each axis lie at rectangle corners.
    // This computes a single analytic zoom, rather than probing a few pixels.
    for (const auto& t : plan.outputToSource)
    {
        if (stopped(cancel)) return fail(error,"Stabilization cancelled");
        const double c = std::abs(std::cos(t.angle)), s = std::abs(std::sin(t.angle));
        const double roomX = halfX-std::abs(t.x), roomY = halfY-std::abs(t.y);
        if (roomX <= 0 || roomY <= 0) return fail(error,"Stabilization translation exceeds the visible extent");
        plan.zoom = std::max({plan.zoom,(c*halfX+s*halfY)/roomX,(s*halfX+c*halfY)/roomY});
    }
    if (!std::isfinite(plan.zoom) || plan.zoom > maxZoom)
        return fail(error,"Stabilization requires more than the allowed border crop");
    result = std::move(plan); error.clear(); return true;
}

namespace
{
size_t sampleBytes (SampleType type)
{
    switch (type) { case SampleType::u8: return 1; case SampleType::u16: return 2; case SampleType::f32: return 4; }
    return 0;
}
bool validPlane (const void* pixels, uint32_t width, uint32_t height, uint32_t channels,
                 size_t stride, size_t bytes, SampleType type)
{
    const size_t sample = sampleBytes(type);
    if (!pixels || !sample || !width || !height || width > maxExtent || height > maxExtent
        || uint64_t(width)*height > maxPixels || !channels || channels > 4) return false;
    const size_t row = size_t(width)*channels*sample;
    return stride >= row && stride <= maxPixels*16 && size_t(height-1) <= (SIZE_MAX-row)/stride
        && bytes >= size_t(height-1)*stride+row;
}
double readSample (const uint8_t* p, SampleType type)
{
    if (type == SampleType::u8) return *p;
    if (type == SampleType::u16) { uint16_t v; std::memcpy(&v,p,2); return v; }
    float v; std::memcpy(&v,p,4); return v;
}
void writeSample (uint8_t* p, SampleType type, double value)
{
    if (type == SampleType::u8) { *p = uint8_t(std::clamp(std::lround(value),0l,255l)); return; }
    if (type == SampleType::u16)
    { const auto v = uint16_t(std::clamp(std::lround(value),0l,65535l)); std::memcpy(p,&v,2); return; }
    const auto v = float(value); std::memcpy(p,&v,4);
}
}
bool warpPlane (const RenderPlan& plan, size_t index, FrameKey key, const ConstPlane& input,
                const MutablePlane& output, std::string& error, const CancelCheck& cancel)
{
    if (!validSource(plan.source,error)) return false;
    if (plan.keys.size() != plan.outputToSource.size() || index >= plan.keys.size()
        || plan.keys[index].ordinal != key.ordinal || plan.keys[index].pts != key.pts)
        return fail(error,"Stabilization render frame does not match analysis");
    if (!std::isfinite(plan.zoom) || plan.zoom < 1 || plan.zoom > 2
        || !finite(plan.outputToSource[index]))
        return fail(error,"Invalid stabilization render plan");
    if (!validPlane(input.pixels,input.width,input.height,input.channels,input.strideBytes,input.byteCount,input.sampleType)
        || !validPlane(output.pixels,output.width,output.height,output.channels,output.strideBytes,output.byteCount,output.sampleType)
        || input.width != output.width || input.height != output.height
        || input.channels != output.channels || input.sampleType != output.sampleType
        || input.width > plan.source.width || input.height > plan.source.height)
        return fail(error,"Invalid stabilization plane storage or extent");
    const auto inputAddress = reinterpret_cast<uintptr_t>(input.pixels);
    const auto outputAddress = reinterpret_cast<uintptr_t>(output.pixels);
    if (input.byteCount > UINTPTR_MAX-inputAddress || output.byteCount > UINTPTR_MAX-outputAddress
        || (inputAddress < outputAddress+output.byteCount && outputAddress < inputAddress+input.byteCount))
        return fail(error,"Stabilization planes overlap");
    const double sar = double(plan.source.sarNumerator)/plan.source.sarDenominator;
    const double sx = double(plan.source.width)/input.width, sy = double(plan.source.height)/input.height;
    const auto t = plan.outputToSource[index];
    const double c = std::cos(t.angle), s = std::sin(t.angle);
    const size_t sample = sampleBytes(input.sampleType);
    const auto* src = static_cast<const uint8_t*>(input.pixels);
    auto* dst = static_cast<uint8_t*>(output.pixels);
    for (uint32_t y = 0; y < output.height; ++y)
    {
        if (stopped(cancel)) return fail(error,"Stabilization cancelled");
        for (uint32_t x = 0; x < output.width; ++x)
        {
            const double px = ((x+0.5)*sx-plan.source.width*0.5)*sar/plan.zoom;
            const double py = ((y+0.5)*sy-plan.source.height*0.5)/plan.zoom;
            double ix = ((c*px-s*py+t.x)/sar+plan.source.width*0.5)/sx-0.5;
            double iy = ((s*px+c*py+t.y)+plan.source.height*0.5)/sy-0.5;
            // Chroma sample centers have a smaller support rectangle. Extension
            // by at most half a sample is the normal edge replication for that
            // plane. Luma corners were admitted by the clip-wide crop above.
            const double margin = sx > 1 || sy > 1 ? 0.5 : 1e-7;
            if (!std::isfinite(ix) || !std::isfinite(iy) || ix < -margin || iy < -margin
                || ix > input.width-1+margin || iy > input.height-1+margin)
                return fail(error,"Stabilization warp falls outside the admitted crop");
            ix = std::clamp(ix,0.0,double(input.width-1)); iy = std::clamp(iy,0.0,double(input.height-1));
            const auto x0 = uint32_t(ix), y0 = uint32_t(iy);
            const auto x1 = std::min(x0+1,input.width-1), y1 = std::min(y0+1,input.height-1);
            const double dx = ix-x0, dy = iy-y0;
            for (uint32_t channel = 0; channel < input.channels; ++channel)
            {
                const auto read = [&](uint32_t ax, uint32_t ay)
                { return readSample(src+size_t(ay)*input.strideBytes+(size_t(ax)*input.channels+channel)*sample,input.sampleType); };
                const double a = read(x0,y0), b = read(x1,y0), d = read(x0,y1), e = read(x1,y1);
                if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(d) || !std::isfinite(e))
                    return fail(error,"Non-finite stabilization source sample");
                const double value = (1-dy)*((1-dx)*a+dx*b)+dy*((1-dx)*d+dx*e);
                writeSample(dst+size_t(y)*output.strideBytes+(size_t(x)*output.channels+channel)*sample,input.sampleType,value);
            }
        }
    }
    error.clear(); return true;
}
} // namespace videohelper::stabilization
