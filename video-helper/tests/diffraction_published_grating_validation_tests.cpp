#include "support/diffraction_reference_oracle.h"
#include "../src/sha256.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace
{
constexpr std::size_t kExpectedSampleCount = 35;
constexpr double kDutyCycle = 0.60;


bool check(bool condition, const char* message)
{
    if (!condition)
        std::cerr << "FAIL: " << message << '\n';
    return condition;
}

double metadataNumber(const std::string& metadata, const std::string& key)
{
    const auto name = metadata.find('"' + key + '"');
    const auto colon = name == std::string::npos ? name : metadata.find(':', name);
    if (colon == std::string::npos)
        throw std::runtime_error("missing metadata number: " + key);
    return std::stod(metadata.substr(colon + 1));
}
}

int main()
{
    using namespace diffractionmaterial;
    using namespace diffractionmaterial::reference;

    std::ifstream fixtureBytes(DIFFRACTION_PUBLISHED_FIXTURE_PATH, std::ios::binary);
    const std::string fixturePayload((std::istreambuf_iterator<char>(fixtureBytes)), {});
    if (!check(fixtureBytes.is_open(), "published grating fixture opens"))
        return 1;

    std::ifstream metadataFile(DIFFRACTION_PUBLISHED_METADATA_PATH);
    const std::string metadata((std::istreambuf_iterator<char>(metadataFile)), {});
    videohelper::Sha256 fixtureHash;
    fixtureHash.update(fixturePayload.data(), fixturePayload.size());
    const auto expectedHash = "bf7b619b97c65abb3002c284a48bf70cb3dc17fed9258af3348c395db8251c1d";
    double maximumRmsError = 0.0;
    double maximumPointError = 0.0;
    try
    {
        maximumRmsError = metadataNumber(metadata, "maximum_rms_absolute_efficiency");
        maximumPointError = metadataNumber(metadata, "maximum_point_absolute_efficiency");
    }
    catch (const std::exception& exception)
    {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
    if (!check(metadataFile.is_open() && fixtureHash.finishHex() == expectedHash
                   && metadata.find(expectedHash) != std::string::npos
                   && metadata.find("10.1364/AO.548315") != std::string::npos
                   && maximumRmsError > 0.0 && maximumPointError > 0.0,
               "published fixture provenance, calibration, uncertainty, and checksum are attached"))
        return 1;

    std::istringstream fixture(fixturePayload);

    std::size_t sampleCount = 0;
    double squaredError = 0.0;
    double maximumError = 0.0;
    std::string line;
    while (std::getline(fixture, line))
    {
        if (line.empty() || line.front() == '#'
            || line.rfind("normalized_depth", 0) == 0)
        {
            continue;
        }

        std::stringstream row(line);
        std::string normalizedDepthText;
        std::string firstOrderAngleText;
        std::string publishedEfficiencyText;
        if (!std::getline(row, normalizedDepthText, ',')
            || !std::getline(row, firstOrderAngleText, ',')
            || !std::getline(row, publishedEfficiencyText, ','))
        {
            std::cerr << "FAIL: malformed published grating fixture row\n";
            return 1;
        }

        const auto normalizedDepth = std::stod(normalizedDepthText);
        const auto firstOrderAngleDegrees = std::stod(firstOrderAngleText);
        const auto publishedEfficiency = std::stod(publishedEfficiencyText);
        if (!check(normalizedDepth >= 0.15 && normalizedDepth <= 0.45,
                   "published normalized depth remains in declared validation range")
            || !check(firstOrderAngleDegrees >= 20.0 && firstOrderAngleDegrees <= 60.0,
                      "published first-order angle remains in declared validation range")
            || !check(publishedEfficiency >= 0.0 && publishedEfficiency <= 1.0,
                      "published efficiency is a physical fraction"))
        {
            return 1;
        }

        const auto predictedEfficiency = profileOrderEfficiency(
            GrooveProfile::BinaryRectangular,
            1.0,
            normalizedDepth,
            kDutyCycle,
            1.0,
            1);
        const auto error = std::abs(predictedEfficiency - publishedEfficiency);
        squaredError += error * error;
        maximumError = std::max(maximumError, error);
        ++sampleCount;
    }

    if (!check(sampleCount == kExpectedSampleCount,
               "published grating fixture sample count is exact"))
    {
        return 1;
    }

    const auto rmsError = std::sqrt(squaredError / static_cast<double>(sampleCount));
    if (!check(rmsError <= maximumRmsError,
               "scalar phase-profile RMS error stays within 4.5 percentage points")
        || !check(maximumError <= maximumPointError,
                  "scalar phase-profile maximum error stays within 7.5 percentage points"))
    {
        return 1;
    }

    std::cout << "Published rectangular-grating validation: " << sampleCount
              << " samples, RMS=" << rmsError
              << ", max=" << maximumError << "\n";
    return 0;
}
