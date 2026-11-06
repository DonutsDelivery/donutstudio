#include "environment_map.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <new>
#include <utility>

namespace videohelper::pathtrace
{
namespace
{

constexpr float kPiF = 3.14159265358979323846f;
constexpr float kTwoPiF = 6.28318530717958647692f;
constexpr double kPiD = 3.14159265358979323846;
constexpr double kTwoPiD = 6.28318530717958647692;
constexpr float kOneBelow = 0.99999994f;  // largest float below 1

// Rec.709 luminance; the decoder writes scene-linear Rec.709 primaries.
PT_SHARED_ENTRY double luminance (double r, double g, double b) noexcept
{
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
}

PT_SHARED_ENTRY float fractOf (float x) noexcept { return x - std::floor (x); }

PT_SHARED_ENTRY float clampEnvironmentUnit (float x) noexcept { return std::min (1.0f, std::max (0.0f, x)); }

PT_SHARED_ENTRY bool finiteNonNegative (float x) noexcept { return std::isfinite (x) && x >= 0.0f; }

// A product past a float keeps its hue at the largest float.
PT_SHARED_ENTRY Vec3 fitToFloat (double x, double y, double z) noexcept
{
    const double peak = std::max (x, std::max (y, z)), limit = std::numeric_limits<float>::max();
    const double fit = peak > limit ? limit / peak : 1.0;
    return { static_cast<float> (x * fit), static_cast<float> (y * fit), static_cast<float> (z * fit) };
}

// GLSL mix: a * (1 - t) + b * t.
PT_SHARED_ENTRY Vec3 mix (Vec3 a, Vec3 b, float t) noexcept { return a * (1.0f - t) + b * t; }

} // namespace

// Host only: the Metal kernel (device_metal.metal) leaves this out.
#if ! defined(PATHTRACE_DEVICE_METAL)
bool EnvironmentPixels::valid() const noexcept
{
    return width > 0 && height > 0 && texels != nullptr
        && texels->size() == static_cast<std::size_t> (width) * static_cast<std::size_t> (height);
}

EnvironmentPixels copyEnvironmentPixels (std::uint32_t width, std::uint32_t height, const float* rgb,
                                         visualenvironmentimage::Identity asset)
{
    EnvironmentPixels pixels;
    if (rgb == nullptr || width == 0 || height == 0)
        return pixels;
    const std::size_t count = static_cast<std::size_t> (width) * static_cast<std::size_t> (height);
    auto texels = std::make_shared<EnvironmentTexels> (count);
    for (std::size_t i = 0; i < count; ++i)
        (*texels)[i] = { rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2] };
    pixels.asset = std::move (asset);
    pixels.width = width;
    pixels.height = height;
    pixels.texels = std::move (texels);
    return pixels;
}

Vec3 environmentImageTint (const HarmonicMIDI::grid::SceneLightRecord& record) noexcept
{
    // Same float expression as the raster uniform.
    const float scale = std::exp2 (record.environmentImage.exposureStops) * record.intensity;
    const Vec3 tint { record.color.x * scale, record.color.y * scale, record.color.z * scale };
    if (isFinite (tint))
        return tint;
    const double gain = std::exp2 (static_cast<double> (record.environmentImage.exposureStops)) * record.intensity;
    return fitToFloat (record.color.x * gain, record.color.y * gain, record.color.z * gain);
}

float environmentImageOrientationRadians (const HarmonicMIDI::grid::SceneLightRecord& record) noexcept
{
    return record.environmentImage.orientationDegrees * 0.01745329251994329577f;
}

void EnvironmentMap::clear() noexcept
{
    ownedTexels_.reset();
    width_ = 0;
    height_ = 0;
    tint_ = {};
    orientation_ = 0.0f;
    ownedMarginal_.clear();
    ownedConditional_.clear();
    ownedCellSolidAngle_.clear();
    ownedRowCos_.clear();
    integrated_ = {};
    maximumTexel_ = {};
    refreshView();
}

void EnvironmentMap::refreshView() noexcept
{
    texels_ = ownedTexels_ != nullptr ? device::BufferView<std::array<float, 3>> { ownedTexels_->data(), ownedTexels_->size() }
                                      : device::BufferView<std::array<float, 3>> {};
    marginal_ = { ownedMarginal_.data(), ownedMarginal_.size() };
    conditional_ = { ownedConditional_.data(), ownedConditional_.size() };
    cellSolidAngle_ = { ownedCellSolidAngle_.data(), ownedCellSolidAngle_.size() };
    rowCos_ = { ownedRowCos_.data(), ownedRowCos_.size() };
}

EnvironmentMap::EnvironmentMap (const EnvironmentMap& other)
    : EnvironmentMapView (other),
      ownedTexels_ (other.ownedTexels_),
      ownedMarginal_ (other.ownedMarginal_),
      ownedConditional_ (other.ownedConditional_),
      ownedCellSolidAngle_ (other.ownedCellSolidAngle_),
      ownedRowCos_ (other.ownedRowCos_)
{
    refreshView();
}

EnvironmentMap::EnvironmentMap (EnvironmentMap&& other) noexcept
    : EnvironmentMapView (other),
      ownedTexels_ (std::move (other.ownedTexels_)),
      ownedMarginal_ (std::move (other.ownedMarginal_)),
      ownedConditional_ (std::move (other.ownedConditional_)),
      ownedCellSolidAngle_ (std::move (other.ownedCellSolidAngle_)),
      ownedRowCos_ (std::move (other.ownedRowCos_))
{
    refreshView();
    other.clear();
}

EnvironmentMap& EnvironmentMap::operator= (const EnvironmentMap& other)
{
    if (this != &other)
    {
        EnvironmentMap copy (other);
        *this = std::move (copy);
    }
    return *this;
}

EnvironmentMap& EnvironmentMap::operator= (EnvironmentMap&& other) noexcept
{
    if (this != &other)
    {
        EnvironmentMapView::operator= (other);
        ownedTexels_ = std::move (other.ownedTexels_);
        ownedMarginal_ = std::move (other.ownedMarginal_);
        ownedConditional_ = std::move (other.ownedConditional_);
        ownedCellSolidAngle_ = std::move (other.ownedCellSolidAngle_);
        ownedRowCos_ = std::move (other.ownedRowCos_);
        refreshView();
        other.clear();
    }
    return *this;
}

bool EnvironmentMap::build (const EnvironmentPixels& pixels, Vec3 tint, float orientationRadians)
{
    clear();
    if (! pixels.valid() || ! finiteNonNegative (tint.x) || ! finiteNonNegative (tint.y) || ! finiteNonNegative (tint.z)
        || ! std::isfinite (orientationRadians))
        return false;

    const std::uint32_t w = pixels.width;
    const std::uint32_t h = pixels.height;
    const EnvironmentTexels& source = *pixels.texels;
    Vec3 maximum {};
    for (const auto& t : source)
    {
        if (! finiteNonNegative (t[0]) || ! finiteNonNegative (t[1]) || ! finiteNonNegative (t[2]))
            return false;
        maximum = vmax (maximum, Vec3 { t[0], t[1], t[2] });
    }

    try
    {
        ownedRowCos_.resize (static_cast<std::size_t> (h) + 1);
        ownedCellSolidAngle_.resize (h);
        ownedMarginal_.resize (static_cast<std::size_t> (h) + 1);
        ownedConditional_.resize (static_cast<std::size_t> (h) * (static_cast<std::size_t> (w) + 1));
    }
    catch (const std::bad_alloc&)
    {
        clear();
        return false;
    }

    for (std::uint32_t j = 0; j <= h; ++j)
        ownedRowCos_[j] = std::cos (kPiD * static_cast<double> (j) / static_cast<double> (h));
    ownedRowCos_[0] = 1.0;
    ownedRowCos_[h] = -1.0;
    for (std::uint32_t j = 0; j < h; ++j)
        ownedCellSolidAngle_[j] = static_cast<float> (kTwoPiD / static_cast<double> (w) * (ownedRowCos_[j] - ownedRowCos_[j + 1]));

    const double tr = tint.x;
    const double tg = tint.y;
    const double tb = tint.z;
    const auto at = [&] (std::uint32_t x, std::uint32_t y) -> const std::array<float, 3>& {
        return source[static_cast<std::size_t> (y) * w + x];
    };

    // Horizontally filtered luminance of one row, and the row's RGB sum (the
    // horizontal kernel wraps, so it keeps the row sum).
    std::vector<double> filtered[3];
    double rowSum[3][3] {};
    for (auto& row : filtered)
        row.resize (w);
    const auto filterRow = [&] (std::uint32_t y, std::vector<double>& out, double* sums) {
        sums[0] = sums[1] = sums[2] = 0.0;
        for (std::uint32_t x = 0; x < w; ++x)
        {
            const auto& t = at (x, y);
            sums[0] += t[0];
            sums[1] += t[1];
            sums[2] += t[2];
        }
        for (std::uint32_t x = 0; x < w; ++x)
        {
            const auto& left = at ((x + w - 1) % w, y);
            const auto& centre = at (x, y);
            const auto& right = at ((x + 1) % w, y);
            const auto lum = [&] (const std::array<float, 3>& t) { return luminance (tr * t[0], tg * t[1], tb * t[2]); };
            out[x] = 0.125 * lum (left) + 0.75 * lum (centre) + 0.125 * lum (right);
        }
    };

    // Rolling window of rows y - 1, y, y + 1 (clamped).
    std::vector<double>* above = &filtered[0];
    std::vector<double>* current = &filtered[1];
    std::vector<double>* below = &filtered[2];
    double* sumAbove = rowSum[0];
    double* sumCurrent = rowSum[1];
    double* sumBelow = rowSum[2];
    filterRow (0, *current, sumCurrent);
    *above = *current;
    std::copy (sumCurrent, sumCurrent + 3, sumAbove);
    if (h > 1)
        filterRow (1, *below, sumBelow);
    else
    {
        *below = *current;
        std::copy (sumCurrent, sumCurrent + 3, sumBelow);
    }

    double integrated[3] {};
    double runningRows = 0.0;
    ownedMarginal_[0] = 0.0f;
    std::vector<double> rowWeight (h);
    for (std::uint32_t y = 0; y < h; ++y)
    {
        float* cdf = &ownedConditional_[static_cast<std::size_t> (y) * (static_cast<std::size_t> (w) + 1)];
        double running = 0.0;
        cdf[0] = 0.0f;
        for (std::uint32_t x = 0; x < w; ++x)
        {
            running += 0.125 * (*above)[x] + 0.75 * (*current)[x] + 0.125 * (*below)[x];
            cdf[x + 1] = static_cast<float> (running);
        }
        if (running > 0.0 && std::isfinite (running))
        {
            const double inverse = 1.0 / running;
            for (std::uint32_t x = 1; x < w; ++x)
                cdf[x] = std::min (1.0f, static_cast<float> (static_cast<double> (cdf[x]) * inverse));
        }
        else
        {
            running = 0.0;
            for (std::uint32_t x = 1; x < w; ++x)
                cdf[x] = static_cast<float> (x) / static_cast<float> (w);
        }
        cdf[w] = 1.0f;
        // Mean luminance of the row's cells times one cell's solid angle,
        // summed over the row.
        rowWeight[y] = running * static_cast<double> (ownedCellSolidAngle_[y]);
        runningRows += rowWeight[y];
        for (int c = 0; c < 3; ++c)
            integrated[c] += (0.125 * sumAbove[c] + 0.75 * sumCurrent[c] + 0.125 * sumBelow[c])
                           * static_cast<double> (ownedCellSolidAngle_[y]);

        // Advance the window.
        std::swap (above, current);
        std::swap (current, below);
        std::swap (sumAbove, sumCurrent);
        std::swap (sumCurrent, sumBelow);
        if (y + 2 < h)
            filterRow (y + 2, *below, sumBelow);
        else
        {
            *below = *current;
            std::copy (sumCurrent, sumCurrent + 3, sumBelow);
        }
    }

    if (! (runningRows > 0.0) || ! std::isfinite (runningRows))
    {
        clear();
        return false;
    }
    double accumulated = 0.0;
    for (std::uint32_t y = 0; y < h; ++y)
    {
        accumulated += rowWeight[y];
        ownedMarginal_[y + 1] = static_cast<float> (accumulated / runningRows);
    }
    ownedMarginal_[h] = 1.0f;

    ownedTexels_ = pixels.texels;
    width_ = w;
    height_ = h;
    tint_ = tint;
    orientation_ = orientationRadians;
    integrated_ = fitToFloat (integrated[0] * tr, integrated[1] * tg, integrated[2] * tb);
    maximumTexel_ = maximum;
    refreshView();
    return true;
}
#endif

PT_SHARED_ENTRY Vec3 EnvironmentMapView::texel (std::uint32_t x, std::uint32_t y) const noexcept
{
    const PT_GLOBAL auto& t = texels_[static_cast<std::size_t> (y) * width_ + x];
    return { t[0], t[1], t[2] };
}

PT_SHARED_ENTRY Vec2 EnvironmentMapView::directionToUv (Vec3 direction) const noexcept
{
    const Vec3 d = normalizeOrZero (direction);
    const float longitude = d.x * d.x + d.z * d.z > 1.0e-12f ? std::atan2 (d.x, d.z) : 0.0f;
    const float u = fractOf ((longitude - orientation_) / kTwoPiF + 0.5f);
    const float v = std::acos (std::min (1.0f, std::max (-1.0f, d.y))) / kPiF;
    return { u, v };
}

PT_SHARED_ENTRY Vec3 EnvironmentMapView::uvToDirection (Vec2 uv) const noexcept
{
    const double longitude = (static_cast<double> (uv.x) - 0.5) * kTwoPiD + static_cast<double> (orientation_);
    const double polar = static_cast<double> (uv.y) * kPiD;
    const double s = std::sin (polar);
    return { static_cast<float> (s * std::sin (longitude)), static_cast<float> (std::cos (polar)),
             static_cast<float> (s * std::cos (longitude)) };
}

PT_SHARED_ENTRY Vec3 EnvironmentMapView::radiance (Vec3 direction) const noexcept
{
    if (! active())
        return {};
    const Vec2 uv = directionToUv (direction);
    const auto w = static_cast<std::int64_t> (width_);
    const auto h = static_cast<std::int64_t> (height_);
    const float px = uv.x * static_cast<float> (width_) - 0.5f;
    const float py = uv.y * static_cast<float> (height_) - 0.5f;
    const float fx = std::floor (px);
    const float fy = std::floor (py);
    const auto ax = static_cast<std::int64_t> (fx);
    const auto ay = static_cast<std::int64_t> (fy);
    const float tx = px - fx;
    const float ty = py - fy;
    const auto x0 = static_cast<std::uint32_t> ((ax % w + w) % w);
    const auto x1 = static_cast<std::uint32_t> ((static_cast<std::int64_t> (x0) + 1) % w);
    const auto y0 = static_cast<std::uint32_t> (std::min (std::max (ay, std::int64_t { 0 }), h - 1));
    const auto y1 = static_cast<std::uint32_t> (std::min (std::max (ay + 1, std::int64_t { 0 }), h - 1));
    const Vec3 top = mix (texel (x0, y0), texel (x1, y0), tx);
    const Vec3 bottom = mix (texel (x0, y1), texel (x1, y1), tx);
    const Vec3 filtered = mix (top, bottom, ty);
    const Vec3 radiance = filtered * tint_;
    if (isFinite (radiance))
        return radiance;
    return fitToFloat (static_cast<double> (filtered.x) * tint_.x, static_cast<double> (filtered.y) * tint_.y,
                       static_cast<double> (filtered.z) * tint_.z);
}

PT_SHARED_ENTRY std::uint32_t EnvironmentMapView::rowOf (Vec3 d) const noexcept
{
    // atan2 keeps precision near the poles, where acos(d.y) does not.
    const double polar = std::atan2 (std::sqrt (static_cast<double> (d.x) * d.x + static_cast<double> (d.z) * d.z),
                                     static_cast<double> (d.y));
    const double row = polar / kPiD * static_cast<double> (height_);
    return static_cast<std::uint32_t> (std::min (std::max (row, 0.0), static_cast<double> (height_ - 1)));
}

PT_SHARED_ENTRY std::uint32_t EnvironmentMapView::columnOf (float u) const noexcept
{
    const float column = u * static_cast<float> (width_);
    return static_cast<std::uint32_t> (std::min (std::max (column, 0.0f), static_cast<float> (width_ - 1)));
}

PT_SHARED_ENTRY float EnvironmentMapView::cellSolidAngle (std::uint32_t row) const noexcept
{
    return row < cellSolidAngle_.size() ? cellSolidAngle_[row] : 0.0f;
}

PT_SHARED_ENTRY float EnvironmentMapView::pdf (Vec3 direction) const noexcept
{
    if (! active())
        return 0.0f;
    const Vec3 d = normalizeOrZero (direction);
    if (dot (d, d) == 0.0f)
        return 0.0f;
    const std::uint32_t row = rowOf (d);
    const std::uint32_t column = columnOf (directionToUv (d).x);
    const float rowProbability = marginal_[row + 1] - marginal_[row];
    const PT_GLOBAL float* cdf = &conditional_[static_cast<std::size_t> (row) * (static_cast<std::size_t> (width_) + 1)];
    const float columnProbability = cdf[column + 1] - cdf[column];
    const float solidAngle = cellSolidAngle_[row];
    if (! (rowProbability > 0.0f) || ! (columnProbability > 0.0f) || ! (solidAngle > 0.0f))
        return 0.0f;
    return rowProbability * columnProbability / solidAngle;
}

PT_SHARED_ENTRY Vec3 EnvironmentMapView::sample (Vec2 u, float& pdfOut) const noexcept
{
    pdfOut = 0.0f;
    if (! active())
        return {};
    const float uRow = std::min (std::max (u.y, 0.0f), float (kOneBelow));
    const float uColumn = std::min (std::max (u.x, 0.0f), float (kOneBelow));

    // Last row whose CDF start is <= uRow; zero-width rows are skipped.
    const auto rowEnd = static_cast<std::ptrdiff_t> (upperBoundIndex (marginal_.data, marginal_.size(), uRow));
    const auto row = static_cast<std::uint32_t> (std::min<std::ptrdiff_t> (std::max<std::ptrdiff_t> (rowEnd - 1, 0), height_ - 1));
    const float r0 = marginal_[row];
    const float rowProbability = marginal_[row + 1] - r0;
    if (! (rowProbability > 0.0f))
        return {};
    const float dv = clampEnvironmentUnit ((uRow - r0) / rowProbability);

    const PT_GLOBAL float* cdf = &conditional_[static_cast<std::size_t> (row) * (static_cast<std::size_t> (width_) + 1)];
    const auto columnEnd = static_cast<std::ptrdiff_t> (upperBoundIndex (cdf, static_cast<std::size_t> (width_) + 1, uColumn));
    const auto column = static_cast<std::uint32_t> (std::min<std::ptrdiff_t> (std::max<std::ptrdiff_t> (columnEnd - 1, 0), width_ - 1));
    const float c0 = cdf[column];
    const float columnProbability = cdf[column + 1] - c0;
    if (! (columnProbability > 0.0f))
        return {};
    const float du = clampEnvironmentUnit ((uColumn - c0) / columnProbability);

    // Uniform in solid angle inside the cell.
    const double cosPolar = rowCos_[row] + static_cast<double> (dv) * (rowCos_[row + 1] - rowCos_[row]);
    const double sinPolar = std::sqrt (std::max (0.0, 1.0 - cosPolar * cosPolar));
    const double uCoordinate = (static_cast<double> (column) + static_cast<double> (du)) / static_cast<double> (width_);
    const double longitude = (uCoordinate - 0.5) * kTwoPiD + static_cast<double> (orientation_);
    const Vec3 direction = normalizeOrZero ({ static_cast<float> (sinPolar * std::sin (longitude)), static_cast<float> (cosPolar),
                                              static_cast<float> (sinPolar * std::cos (longitude)) });
    const float solidAngle = cellSolidAngle_[row];
    if (dot (direction, direction) == 0.0f || ! (solidAngle > 0.0f))
        return {};
    pdfOut = rowProbability * columnProbability / solidAngle;
    return direction;
}

} // namespace videohelper::pathtrace
