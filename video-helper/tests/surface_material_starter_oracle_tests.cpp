#include "support/surface_material_starter_oracle.h"
#include "../src/sha256.h"

#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#ifndef SURFACE_MATERIAL_STARTER_ORACLE_PATH
#error "SURFACE_MATERIAL_STARTER_ORACLE_PATH must name the generated oracle"
#endif

namespace
{
bool rejectsSchema (const nlohmann::json& source, const nlohmann::json& schema)
{
    auto candidate = source;
    candidate["schema"] = schema;
    try
    {
        static_cast<void> (surfacematerialstarteroracle::loadDocument (candidate));
    }
    catch (const std::runtime_error& error)
    {
        return std::string (error.what()) == "unsupported Surface Material oracle schema";
    }
    return false;
}
}

int main()
{
    try
    {
        std::ifstream stream (SURFACE_MATERIAL_STARTER_ORACLE_PATH, std::ios::binary);
        if (! stream)
            throw std::runtime_error ("cannot open generated Surface Material oracle");
        const auto document = nlohmann::json::parse (stream);
        const auto oracle = surfacematerialstarteroracle::loadDocument (document);

        std::vector<std::uint8_t> canonicalBytes;
        canonicalBytes.reserve (oracle.materials.size() * 4);
        for (const auto& material : oracle.materials)
            canonicalBytes.insert (canonicalBytes.end(), material.centerRgba8.begin(), material.centerRgba8.end());

        char fnvText[17] {};
        std::snprintf (fnvText, sizeof (fnvText), "%016llx",
                       static_cast<unsigned long long> (surfacematerialstarteroracle::fnv1a64 (canonicalBytes)));
        videohelper::Sha256 sha256;
        sha256.update (canonicalBytes.data(), canonicalBytes.size());

        const bool schemasFailClosed = rejectsSchema (document, 2)
            && rejectsSchema (document, 4)
            && rejectsSchema (document, "3")
            && rejectsSchema (document, nullptr);
        if (canonicalBytes.size() != 20 || oracle.sampledRgba8Fnv1a64 != fnvText
            || oracle.sampledRgba8Sha256 != sha256.finishHex() || ! schemasFailClosed)
        {
            std::cerr << "Surface Material oracle schema or cross-language digest mismatch\n";
            return 1;
        }

        std::cout << "Surface Material oracle schema and cross-language digest PASS\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}