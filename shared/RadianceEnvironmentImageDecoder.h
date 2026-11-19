#pragma once

#include "VisualEnvironmentImageAssetContract.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

namespace visualenvironmentimage
{
namespace radiance_detail
{
inline bool numbers(std::string_view text, double* values, std::size_t count)
{
    std::istringstream input{std::string(text)};
    input.imbue(std::locale::classic());
    for (std::size_t i = 0; i < count; ++i)
        if (!(input >> values[i]) || !std::isfinite(values[i])) return false;
    input >> std::ws;
    return input.eof();
}
inline bool decimal(const std::string& text, std::uint32_t& value)
{
    if (text.empty()) return false;
    std::uint64_t parsed = 0;
    for (char c : text)
    {
        if (c < '0' || c > '9') return false;
        parsed = parsed * 10 + static_cast<unsigned>(c - '0');
        if (parsed > kMaxDimension) return false;
    }
    value = static_cast<std::uint32_t>(parsed);
    return value > 0;
}
} // namespace radiance_detail

// Independent implementation of the Radiance picture specification, not copied
// Radiance source. See artifacts/planning/missing-features/evidence/
// plan3-environment-image-asset-authority-20261001.md for primary format sources.
inline bool decodeRadianceEnvironment(const std::uint8_t* bytes, std::size_t size,
                                      DecodedImage& output, std::string& error)
{
    output = {};
    error.clear();
    const auto fail = [&](const char* message) { error = message; output = {}; return false; };
    if (bytes == nullptr || size == 0 || size > kMaxSourceBytes)
        return fail("HDR environment source size exceeds the decoder limit");
    std::size_t cursor = 0;
    const auto line = [&](std::string& result)
    {
        result.clear();
        while (cursor < size && bytes[cursor] != '\n')
        {
            const auto c = bytes[cursor++];
            if (result.size() == 512 || c < 0x20 || c > 0x7e) return false;
            result.push_back(static_cast<char>(c));
        }
        if (cursor == size || cursor >= 65536) return false;
        ++cursor;
        return true;
    };
    std::string text;
    if (!line(text) || (text != "#?RADIANCE" && text != "#?RGBE"))
        return fail("HDR environment requires a Radiance RGBE header");
    bool format = false, primariesSeen = false;
    double exposure = 1.0;
    std::array<double,3> correction{1.0,1.0,1.0};
    Metadata metadata;
    while (true)
    {
        if (!line(text)) return fail("HDR environment header is malformed or oversized");
        if (text.empty()) break;
        if (text.front() == '#' || text.find('=') == std::string::npos) continue;
        const auto separator = text.find('=');
        const auto name = text.substr(0, separator);
        const auto value = std::string_view(text).substr(separator + 1);
        if (name == "FORMAT")
        {
            if (format || value != "32-bit_rle_rgbe")
                return fail("HDR environment format is unsupported or duplicated");
            format = true;
        }
        else if (name == "EXPOSURE" || name == "COLORCORR" || name == "PIXASPECT")
        {
            double values[3]{};
            const std::size_t count = name == "COLORCORR" ? 3 : 1;
            if (!radiance_detail::numbers(value, values, count))
                return fail("HDR environment correction metadata is malformed");
            for (std::size_t i = 0; i < count; ++i)
                if (!(values[i] > 0.0)) return fail("HDR environment correction must be positive");
            if (name == "PIXASPECT")
            {
                if (values[0] != 1.0) return fail("HDR environment requires square pixels");
            }
            else if (name == "EXPOSURE") exposure *= values[0];
            else for (std::size_t i = 0; i < 3; ++i) correction[i] *= values[i];
            if (!std::isfinite(exposure) || exposure < 1e-12 || exposure > 1e12)
                return fail("HDR environment cumulative exposure exceeds the decoder limit");
            for (double c : correction)
                if (!std::isfinite(c) || c < 1e-12 || c > 1e12)
                    return fail("HDR environment cumulative color correction exceeds the decoder limit");
        }
        else if (name == "PRIMARIES")
        {
            double p[8]{};
            if (primariesSeen || !radiance_detail::numbers(value, p, 8))
                return fail("HDR environment primaries are malformed or duplicated");
            primariesSeen = true;
            constexpr double rec709[]{.64,.33,.30,.60,.15,.06,.3127,.3290};
            constexpr double radiance[]{.64,.33,.29,.60,.15,.06,1.0/3.0,1.0/3.0};
            const auto matches = [&](const double* expected)
            {
                for (std::size_t i = 0; i < 8; ++i)
                    if (std::abs(p[i] - expected[i]) > 1e-9) return false;
                return true;
            };
            if (matches(rec709)) metadata.sourcePrimaries = SourcePrimaries::Rec709D65;
            else if (!matches(radiance)) return fail("HDR environment primaries are unsupported");
        }
        else if (name != "SOFTWARE" && name != "CAPDATE" && name != "GMT")
            return fail("HDR environment contains unsupported interpretation metadata");
    }
    if (!format || !line(text)) return fail("HDR environment has no RGBE format or resolution");
    std::istringstream resolution(text);
    resolution.imbue(std::locale::classic());
    std::string major, minor, majorSize, minorSize, extra;
    std::uint32_t scans = 0, length = 0;
    if (!(resolution >> major >> majorSize >> minor >> minorSize) || (resolution >> extra)
        || major.size() != 2 || minor.size() != 2
        || (major[0] != '+' && major[0] != '-') || (minor[0] != '+' && minor[0] != '-')
        || (major[1] != 'X' && major[1] != 'Y') || (minor[1] != 'X' && minor[1] != 'Y')
        || major[1] == minor[1] || !radiance_detail::decimal(majorSize, scans)
        || !radiance_detail::decimal(minorSize, length))
        return fail("HDR environment resolution or axis ordering is malformed");
    metadata.width = major[1] == 'X' ? scans : length;
    metadata.height = major[1] == 'Y' ? scans : length;
    if (!valid(metadata)) return fail("HDR environment dimensions exceed the equirectangular limit");
    output.metadata = metadata;
    output.pixels.resize(std::size_t(metadata.width) * metadata.height);
    std::vector<std::array<std::uint8_t,4>> scanline(length);
    const auto readPixel = [&](std::array<std::uint8_t,4>& pixel)
    {
        if (size - cursor < 4) return false;
        std::memcpy(pixel.data(), bytes + cursor, 4); cursor += 4; return true;
    };
    for (std::uint32_t scan = 0; scan < scans; ++scan)
    {
        std::array<std::uint8_t,4> first{};
        if (!readPixel(first)) return fail("HDR environment scanline is truncated");
        if (length >= 8 && first[0] == 2 && first[1] == 2 && first[2] < 128)
        {
            if ((std::uint32_t(first[2]) * 256 + first[3]) != length)
                return fail("HDR environment RLE scanline length does not match");
            for (std::size_t channel = 0; channel < 4; ++channel)
            {
                std::uint32_t position = 0;
                while (position < length)
                {
                    if (cursor == size) return fail("HDR environment RLE is truncated");
                    const auto code = bytes[cursor++];
                    const auto count = std::uint32_t(code > 128 ? code - 128 : code);
                    if (count == 0 || count > length - position)
                        return fail("HDR environment RLE run exceeds the scanline");
                    if (code > 128)
                    {
                        if (cursor == size) return fail("HDR environment RLE run is truncated");
                        const auto value = bytes[cursor++];
                        for (std::uint32_t i = 0; i < count; ++i) scanline[position++][channel] = value;
                    }
                    else
                    {
                        if (count > size - cursor) return fail("HDR environment RLE literals are truncated");
                        for (std::uint32_t i = 0; i < count; ++i) scanline[position++][channel] = bytes[cursor++];
                    }
                }
            }
        }
        else
        {
            std::uint32_t position = 0;
            unsigned shift = 0;
            auto pixel = first;
            while (position < length)
            {
                if (pixel[0] == 1 && pixel[1] == 1 && pixel[2] == 1)
                {
                    if (position == 0 || shift >= 32)
                        return fail("HDR environment old RLE repeat is malformed");
                    const auto count = std::uint64_t(pixel[3]) << shift;
                    if (count > length - position) return fail("HDR environment old RLE run exceeds the scanline");
                    for (std::uint64_t i = 0; i < count; ++i)
                    {
                        scanline[position] = scanline[position - 1]; ++position;
                    }
                    shift += 8;
                }
                else { scanline[position++] = pixel; shift = 0; }
                if (position < length && !readPixel(pixel)) return fail("HDR environment pixels are truncated");
            }
        }
        for (std::uint32_t position = 0; position < length; ++position)
        {
            const auto& rgbe = scanline[position];
            std::array<double,3> rgb{};
            if (rgbe[3] != 0)
            {
                if (std::max({rgbe[0],rgbe[1],rgbe[2]}) < 128)
                    return fail("HDR environment RGBE pixel is unnormalized");
                const auto scale = std::ldexp(1.0, int(rgbe[3]) - 136);
                for (std::size_t c = 0; c < 3; ++c)
                    rgb[c] = (double(rgbe[c]) + .5) * scale / exposure / correction[c];
            }
            if (metadata.sourcePrimaries == SourcePrimaries::RadianceEqualEnergy)
            {
                // RGB -> XYZ from Radiance nominal CRT primaries/EE white,
                // then XYZ -> Rec.709/D65. No chromatic adaptation of radiance.
                rgb = {1.2467412505963913 * rgb[0] - .04176524655496619 * rgb[1],
                       .94827892263401625 * rgb[1],
                       .01118341205337184 * rgb[1] + .89744122299752405 * rgb[2]};
            }
            std::array<float,3> converted{};
            for (std::size_t c = 0; c < 3; ++c)
            {
                if (!std::isfinite(rgb[c]) || rgb[c] < 0)
                    return fail("HDR environment radiance is outside finite nonnegative Rec.709 storage");
                converted[c] = static_cast<float>(std::min(rgb[c], static_cast<double>(std::numeric_limits<float>::max())));
                if (rgb[c] > 0.0 && converted[c] == 0.0f)
                    return fail("HDR environment radiance is below representable RGB32F storage");
            }
            const auto coordinate = [](const std::string& axis, std::uint32_t index, std::uint32_t extent)
            { return axis[0] == '+' ? index : extent - 1 - index; };
            const auto a = coordinate(major, scan, scans), b = coordinate(minor, position, length);
            const auto x = major[1] == 'X' ? a : b;
            const auto y = major[1] == 'Y' ? a : b;
            output.pixels[std::size_t(metadata.height - 1 - y) * metadata.width + x] = converted;
        }
    }
    if (cursor != size) return fail("HDR environment has trailing source bytes");
    return true;
}
} // namespace visualenvironmentimage
