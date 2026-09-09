#pragma once

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace surfacematerialstarteroracle
{
inline constexpr int kSchemaVersion = 3;
inline constexpr std::uint64_t kFnv1a64OffsetBasis = 14695981039346656037ull;
inline constexpr std::uint64_t kFnv1a64Prime = 1099511628211ull;

struct Material final
{
    std::string id;
    std::array<float, 20> pbr {};
    std::array<std::uint8_t, 4> centerRgba8 {};
    std::vector<std::array<std::uint8_t, 4>> rejectedAlternatives;
};

struct Oracle final
{
    std::vector<Material> materials;
    std::string sampledRgba8Sha256;
    std::string sampledRgba8Fnv1a64;
};

inline Oracle loadDocument (const nlohmann::json& root)
{
    const auto schema = root.find ("schema");
    if (schema == root.end() || ! schema->is_number_integer()
        || schema->get<int>() != kSchemaVersion)
        throw std::runtime_error ("unsupported Surface Material oracle schema");

    Oracle result;
    result.sampledRgba8Sha256 = root.at ("sampledRgba8Sha256").get<std::string>();
    result.sampledRgba8Fnv1a64 = root.at ("sampledRgba8Fnv1a64").get<std::string>();
    for (const auto& source : root.at ("materials"))
    {
        Material material;
        material.id = source.at ("id").get<std::string>();
        const auto pbr = source.at ("pbrParameterBlock").get<std::vector<float>>();
        const auto center = source.at ("directionalLitCenterRgba8").get<std::vector<int>>();
        if (pbr.size() != material.pbr.size() || center.size() != material.centerRgba8.size())
            throw std::runtime_error ("malformed Surface Material oracle row");
        std::copy (pbr.begin(), pbr.end(), material.pbr.begin());
        for (std::size_t index = 0; index < center.size(); ++index)
            material.centerRgba8[index] = static_cast<std::uint8_t> (center[index]);
        for (const auto& alternative : source.at ("rejectedWrongShadingRgba8"))
        {
            const auto channels = alternative.get<std::vector<int>>();
            if (channels.size() != 4)
                throw std::runtime_error ("malformed wrong-shading oracle alternative");
            material.rejectedAlternatives.push_back ({
                static_cast<std::uint8_t> (channels[0]),
                static_cast<std::uint8_t> (channels[1]),
                static_cast<std::uint8_t> (channels[2]),
                static_cast<std::uint8_t> (channels[3])
            });
        }
        result.materials.push_back (std::move (material));
    }
    if (result.materials.size() != 5)
        throw std::runtime_error ("Surface Material oracle must contain five rows");
    return result;
}

inline Oracle load (const char* path)
{
    std::ifstream stream (path, std::ios::binary);
    if (! stream)
        throw std::runtime_error (std::string ("cannot open Surface Material oracle: ") + path);
    return loadDocument (nlohmann::json::parse (stream));
}

inline std::uint64_t fnv1a64 (const std::vector<std::uint8_t>& bytes)
{
    auto value = kFnv1a64OffsetBasis;
    for (const auto byte : bytes)
    {
        value ^= byte;
        value *= kFnv1a64Prime;
    }
    return value;
}
} // namespace surfacematerialstarteroracle
