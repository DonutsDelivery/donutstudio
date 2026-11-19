#pragma once

#include "ShaderCatalogManifest.h"
#include "PortableFloatingPointChars.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <string_view>

namespace shadercatalog
{
inline constexpr std::size_t maxParameterWireBytes = 8192;

inline const Parameter& parameterAt(const Entry& entry, std::size_t index)
{
    return parameters[entry.parameterOffset + index];
}

inline std::string runtimeNodeAlias(std::int64_t nodeId)
{
    return nodeId > 0 ? "visual" + std::to_string(nodeId) : std::string {};
}

inline std::string runtimeParameterId(int clipId, std::int64_t nodeId,
                                      std::string_view parameterId)
{
    if (clipId < 0 || nodeId <= 0 || parameterId.empty()
        || parameterId.find('/') != std::string_view::npos)
        return {};
    return "clip" + std::to_string(clipId) + "/" + runtimeNodeAlias(nodeId)
        + "/" + std::string(parameterId);
}

inline bool parseRuntimeNodeAlias(std::string_view alias, int& nodeId)
{
    constexpr std::string_view prefix = "visual";
    if (alias.substr(0, prefix.size()) != prefix || alias.size() == prefix.size())
        return false;
    int parsed = 0;
    const auto* first = alias.data() + prefix.size();
    const auto* last = alias.data() + alias.size();
    const auto result = std::from_chars(first, last, parsed);
    if (result.ec != std::errc {} || result.ptr != last || parsed <= 0)
        return false;
    nodeId = parsed;
    return true;
}

inline bool validParameterName(std::string_view name)
{
    if (name.empty()) return false;
    const auto isAlpha = [](char c)
    {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
    };
    const auto isDigit = [](char c) { return c >= '0' && c <= '9'; };
    if (!isAlpha(name.front())) return false;
    return std::all_of(name.begin() + 1, name.end(),
        [&](char c) { return isAlpha(c) || isDigit(c); });
}

inline bool validateParameterMetadata(const Parameter* source, std::size_t count,
                                      std::string& error)
{
    if (source == nullptr && count != 0)
    {
        error = "FlatShaderBridge parameter metadata is missing";
        return false;
    }
    std::set<std::string_view> names;
    for (std::size_t index = 0; index < count; ++index)
    {
        const auto& parameter = source[index];
        const bool scalar = parameter.type == "float" || parameter.type == "bool"
            || parameter.type == "long";
        const std::size_t expectedComponents = parameter.type == "point2D" ? 2
            : parameter.type == "color" ? 4 : scalar ? 1 : 0;
        if (!validParameterName(parameter.name) || expectedComponents == 0
            || parameter.componentCount != expectedComponents
            || ! names.insert(parameter.name).second)
        {
            error = "FlatShaderBridge parameter metadata is malformed or contains duplicate IDs";
            return false;
        }
        for (std::size_t component = 0; component < parameter.componentCount; ++component)
        {
            const auto defaultValue = parameter.defaultValue[component];
            const auto minimum = parameter.minValue[component];
            const auto maximum = parameter.maxValue[component];
            if (!std::isfinite(defaultValue)
                || (parameter.hasMin && !std::isfinite(minimum))
                || (parameter.hasMax && !std::isfinite(maximum))
                || (parameter.hasMin && parameter.hasMax && minimum > maximum)
                || (parameter.hasMin && defaultValue < minimum)
                || (parameter.hasMax && defaultValue > maximum)
                || (parameter.type == "bool" && defaultValue != 0.0 && defaultValue != 1.0)
                || (parameter.type == "long" && std::floor(defaultValue) != defaultValue))
            {
                error = "FlatShaderBridge parameter metadata has invalid ranges or defaults";
                return false;
            }
        }
    }
    error.clear();
    return true;
}

inline bool validateParameterMetadata(const Entry& entry, std::string& error)
{
    constexpr auto parameterCount = sizeof(parameters) / sizeof(parameters[0]);
    if (entry.parameterOffset > parameterCount
        || entry.parameterCount > parameterCount - entry.parameterOffset)
    {
        error = "FlatShaderBridge parameter metadata is outside the catalog";
        return false;
    }
    return validateParameterMetadata(parameters + entry.parameterOffset,
                                     entry.parameterCount, error);
}

inline std::string parameterComponentKey(const Parameter& parameter, std::size_t component)
{
    if (parameter.componentCount == 1) return std::string(parameter.name);
    static constexpr const char* pointSuffix[] = { ".x", ".y" };
    static constexpr const char* colorSuffix[] = { ".r", ".g", ".b", ".a" };
    return std::string(parameter.name)
        + (parameter.type == "point2D" ? pointSuffix[component] : colorSuffix[component]);
}

inline bool hasParameterComponent(const Entry& entry, std::string_view componentId)
{
    for (std::size_t index = 0; index < entry.parameterCount; ++index)
    {
        const auto& parameter = parameterAt(entry, index);
        for (std::size_t component = 0; component < parameter.componentCount; ++component)
            if (parameterComponentKey(parameter, component) == componentId)
                return true;
    }
    return false;
}

namespace parameter_detail
{
inline std::string numberToWire(double value)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10)
           << (value == 0.0 ? 0.0 : value);
    return stream.str();
}

inline bool parseFloat(std::string_view text, double& value)
{
    if (text.empty()) return false;
    std::size_t position = text.front() == '-' ? 1 : 0;
    if (position == text.size()) return false;
    bool digits = false;
    while (position < text.size() && text[position] >= '0' && text[position] <= '9')
    {
        digits = true;
        ++position;
    }
    if (position < text.size() && text[position] == '.')
    {
        ++position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9')
        {
            digits = true;
            ++position;
        }
    }
    if (!digits) return false;
    if (position < text.size() && (text[position] == 'e' || text[position] == 'E'))
    {
        ++position;
        if (position < text.size() && (text[position] == '+' || text[position] == '-')) ++position;
        const auto exponentStart = position;
        while (position < text.size() && text[position] >= '0' && text[position] <= '9') ++position;
        if (position == exponentStart) return false;
    }
    if (position != text.size()) return false;
    const auto result = portablefloat::fromChars(text.data(), text.data() + text.size(), value,
                                                 std::chars_format::general);
    return result.ec == std::errc{} && result.ptr == text.data() + text.size()
        && std::isfinite(value)
        && std::abs(value) <= static_cast<double>(std::numeric_limits<float>::max());
}

inline bool parseLong(std::string_view text, double& value)
{
    if (text.empty()) return false;
    std::size_t position = text.front() == '-' ? 1 : 0;
    if (position == text.size()) return false;
    for (; position < text.size(); ++position)
        if (text[position] < '0' || text[position] > '9') return false;
    long long parsed = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), parsed, 10);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()
        || parsed < std::numeric_limits<int>::min()
        || parsed > std::numeric_limits<int>::max())
        return false;
    value = static_cast<double>(parsed);
    return true;
}
}

inline std::string parameterWire(const Entry& entry,
                                 const std::map<std::string, double>& values,
                                 std::string& error)
{
    if (!validateParameterMetadata(entry, error)) return {};
    std::string wire;
    for (std::size_t index = 0; index < entry.parameterCount; ++index)
    {
        const auto& parameter = parameterAt(entry, index);
        if (index != 0) wire.push_back(';');
        wire.append(parameter.name);
        wire.push_back(':');
        wire.append(parameter.type);
        wire.push_back('=');
        for (std::size_t component = 0; component < parameter.componentCount; ++component)
        {
            if (component != 0) wire.push_back(',');
            const auto found = values.find(parameterComponentKey(parameter, component));
            if (found == values.end())
            {
                error = "FlatShaderBridge parameter list is incomplete";
                return {};
            }
            const auto value = found->second;
            if (!std::isfinite(value)
                || (parameter.hasMin && value < parameter.minValue[component])
                || (parameter.hasMax && value > parameter.maxValue[component])
                || (parameter.type == "bool" && value != 0.0 && value != 1.0)
                || (parameter.type == "long" && value != std::trunc(value)))
            {
                error = "FlatShaderBridge parameter value has the wrong type or is outside the catalog range";
                return {};
            }
            if (parameter.type == "bool") wire.append(value == 0.0 ? "false" : "true");
            else if (parameter.type == "long") wire.append(std::to_string(static_cast<long long>(value)));
            else wire.append(parameter_detail::numberToWire(value));
        }
    }
    if (values.size() != [&entry]
        {
            std::size_t count = 0;
            for (std::size_t index = 0; index < entry.parameterCount; ++index)
                count += parameterAt(entry, index).componentCount;
            return count;
        }())
    {
        error = "FlatShaderBridge parameter list contains duplicate or unknown values";
        return {};
    }
    error.clear();
    return wire;
}

inline std::string defaultParameterWire(const Entry& entry)
{
    std::string error;
    if (!validateParameterMetadata(entry, error)) return {};
    std::map<std::string, double> values;
    for (std::size_t index = 0; index < entry.parameterCount; ++index)
    {
        const auto& parameter = parameterAt(entry, index);
        for (std::size_t component = 0; component < parameter.componentCount; ++component)
            values.emplace(parameterComponentKey(parameter, component), parameter.defaultValue[component]);
    }
    return parameterWire(entry, values, error);
}

inline bool validateParameterWire(const Entry& entry, std::string_view wire,
                                  std::map<std::string, double>& values,
                                  std::string& error)
{
    if (!validateParameterMetadata(entry, error)) return false;
    values.clear();
    if (entry.parameterCount > maxParameterCount || wire.size() > maxParameterWireBytes)
    {
        error = "FlatShaderBridge parameter list exceeds its bounded catalog transport";
        return false;
    }
    std::size_t cursor = 0;
    for (std::size_t index = 0; index < entry.parameterCount; ++index)
    {
        const auto& parameter = parameterAt(entry, index);
        const auto end = wire.find(';', cursor);
        const auto item = wire.substr(cursor, end == std::string_view::npos ? wire.size() - cursor
                                                                            : end - cursor);
        const auto colon = item.find(':');
        const auto equals = item.find('=');
        if (colon == std::string_view::npos || equals == std::string_view::npos || colon >= equals)
        {
            error = "FlatShaderBridge parameter item syntax is malformed";
            return false;
        }
        if (item.substr(0, colon) != parameter.name)
        {
            error = "FlatShaderBridge parameter list is not the exact catalog order";
            return false;
        }
        if (item.substr(colon + 1, equals - colon - 1) != parameter.type)
        {
            error = "FlatShaderBridge parameter type does not match the catalog manifest";
            return false;
        }

        auto components = item.substr(equals + 1);
        for (std::size_t component = 0; component < parameter.componentCount; ++component)
        {
            const auto comma = components.find(',');
            const auto token = components.substr(0, comma);
            double value = 0.0;
            bool parsed = false;
            if (parameter.type == "bool")
            {
                if (token == "false") { value = 0.0; parsed = true; }
                else if (token == "true") { value = 1.0; parsed = true; }
            }
            else if (parameter.type == "long")
                parsed = parameter_detail::parseLong(token, value);
            else
                parsed = parameter_detail::parseFloat(token, value);
            if (!parsed)
            {
                error = "FlatShaderBridge parameter value has the wrong type or is nonfinite";
                return false;
            }
            if ((parameter.hasMin && value < parameter.minValue[component])
                || (parameter.hasMax && value > parameter.maxValue[component]))
            {
                error = "FlatShaderBridge parameter value is outside the catalog range";
                return false;
            }
            values.emplace(parameterComponentKey(parameter, component), value);
            if (component + 1 < parameter.componentCount)
            {
                if (comma == std::string_view::npos)
                {
                    error = "FlatShaderBridge parameter component count is wrong";
                    return false;
                }
                components.remove_prefix(comma + 1);
            }
            else if (comma != std::string_view::npos)
            {
                error = "FlatShaderBridge parameter component count is wrong";
                return false;
            }
        }
        if (end == std::string_view::npos)
        {
            cursor = wire.size();
            if (index + 1 != entry.parameterCount)
            {
                error = "FlatShaderBridge parameter list is incomplete";
                return false;
            }
        }
        else
        {
            if (index + 1 == entry.parameterCount)
            {
                error = end + 1 == wire.size()
                    ? "FlatShaderBridge parameter list contains a trailing separator"
                    : "FlatShaderBridge parameter list contains duplicate or unknown values";
                return false;
            }
            cursor = end + 1;
        }
    }
    if (cursor != wire.size() || (entry.parameterCount == 0 && !wire.empty()))
    {
        error = "FlatShaderBridge parameter list contains duplicate or unknown values";
        return false;
    }
    error.clear();
    return true;
}
} // namespace shadercatalog
