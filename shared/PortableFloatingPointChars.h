#pragma once

#include <charconv>
#include <cstring>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <type_traits>

namespace portablefloat
{
template <typename T>
inline std::from_chars_result fromChars(const char* first, const char* last, T& value,
                                        std::chars_format format = std::chars_format::general) noexcept
{
    static_assert(std::is_floating_point_v<T>);
#if defined(__APPLE__)
    // libc++ floating-point from_chars requires macOS 26 even with a C++17 SDK.
    (void) format;
    if (first == last || *first == '+') return { first, std::errc::invalid_argument };
    try
    {
        std::istringstream stream(std::string(first, last));
        stream.imbue(std::locale::classic());
        stream >> std::noskipws >> value;
        return !stream.fail() && stream.eof()
            ? std::from_chars_result { last, std::errc {} }
            : std::from_chars_result { first, std::errc::invalid_argument };
    }
    catch (...)
    {
        return { first, std::errc::invalid_argument };
    }
#else
    return std::from_chars(first, last, value, format);
#endif
}

inline std::to_chars_result toChars(char* first, char* last, double value,
                                    std::chars_format format = std::chars_format::general,
                                    int precision = -1)
{
#if defined(__APPLE__)
    // libc++ floating-point to_chars requires macOS 13.3; keep the wire locale fixed.
    (void) format;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(precision >= 0 ? precision
                                              : std::numeric_limits<double>::max_digits10)
           << value;
    const auto text = stream.str();
    if (text.size() > static_cast<std::size_t>(last - first))
        return { last, std::errc::value_too_large };
    std::memcpy(first, text.data(), text.size());
    return { first + text.size(), std::errc {} };
#else
    return precision >= 0 ? std::to_chars(first, last, value, format, precision)
                          : std::to_chars(first, last, value, format);
#endif
}
} // namespace portablefloat
