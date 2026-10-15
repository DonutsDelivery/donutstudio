#include "../../shared/VisualVolumeOperationContract.h"

#include <algorithm>
#include <iostream>
#include <limits>

int main()
{
    int failures = 0;
    const auto check = [&](bool ok, const char* message)
        { if (!ok) { std::cerr << "FAIL: " << message << '\n'; ++failures; } };
    visualvolume::Operation operation;
    operation.sourceStableId = 12; operation.renderStableId = 17;
    const auto encoded = visualvolume::encode(operation);
    visualvolume::Operation decoded;
    check(!encoded.empty() && visualvolume::decode(encoded, decoded)
          && visualvolume::encode(decoded) == encoded, "exact volume operation round trip");
    check(!visualvolume::decode(encoded + "00", decoded), "extra payload bytes reject");
    check(!visualvolume::decode(encoded.substr(0, encoded.size() - 1), decoded), "truncated payload rejects");
    auto malformed = encoded; malformed.back() = 'G';
    check(!visualvolume::decode(malformed, decoded), "noncanonical hexadecimal rejects");
    auto zero = operation; zero.phase = -0.0f;
    const auto canonicalZero = visualvolume::encode(zero);
    zero.phase = 0.0f;
    check(visualvolume::encode(zero) == canonicalZero, "negative zero has one canonical encoding");
    auto negativeZeroWire = canonicalZero;
    negativeZeroWire.replace(negativeZeroWire.size() - 8, 8, "80000000");
    check(!visualvolume::decode(negativeZeroWire, decoded), "noncanonical negative-zero wire rejects");
    for (const auto radius : {0.0f, -1.0f, 0.76f, std::numeric_limits<float>::quiet_NaN()})
    {
        auto invalid = operation; invalid.radius = radius;
        check(visualvolume::encode(invalid).empty(), "invalid authored radius rejects before data generation");
    }
    auto invalid = operation; invalid.resolution = visualvolume::maximumResolution + 1;
    videowire::DenseVolumeDescriptor unchanged;
    unchanged.voxels = {123};
    check(!visualvolume::makeDensity(invalid, unchanged) && unchanged.voxels == std::vector<std::uint8_t>{123},
          "voxel limit rejects without overwriting output");
    invalid = operation; invalid.sourceStableId = invalid.renderStableId;
    check(!visualvolume::valid(invalid), "source and render identities cannot alias");
    videowire::DenseVolumeDescriptor source, repeated, quiet, smaller;
    check(visualvolume::makeDensity(operation, source) && visualvolume::makeDensity(operation, repeated)
          && source.voxels == repeated.voxels, "identical parameters produce identical bounded density bytes");
    check(source.voxels.size() == 32u * 32u * 32u && source.voxels.front() == 0
          && *std::max_element(source.voxels.begin(), source.voxels.end()) > 100,
          "sphere density occupies the interior, not its bounding-box corners");
    auto altered = operation; altered.density = 0;
    check(visualvolume::makeDensity(altered, quiet)
          && std::all_of(quiet.voxels.begin(), quiet.voxels.end(), [](auto value) { return value == 0; }),
          "zero density removes the volume rather than changing an unrelated colour");
    altered = operation; altered.radius = 0.2f;
    check(visualvolume::makeDensity(altered, smaller)
          && std::count_if(smaller.voxels.begin(), smaller.voxels.end(), [](auto value) { return value > 0; })
             < std::count_if(source.voxels.begin(), source.voxels.end(), [](auto value) { return value > 0; }),
          "radius edit changes occupied volume");
    for (const auto resolution : {8u, visualvolume::maximumResolution})
    {
        auto boundary = operation; boundary.resolution = resolution;
        check(visualvolume::makeDensity(boundary, repeated)
              && repeated.voxels.size() == static_cast<std::size_t>(resolution) * resolution * resolution
              && repeated.voxels.size() <= visualvolume::maximumVoxels,
              "both resolution boundaries generate only their admitted voxel budget");
    }
    altered = operation; altered.phase = 1.0f;
    check(visualvolume::makeDensity(altered, repeated) && repeated.voxels != source.voxels,
          "noise phase changes density data");
    altered = operation; altered.noiseFrequency = 2.0f;
    check(visualvolume::makeDensity(altered, repeated) && repeated.voxels != source.voxels,
          "noise frequency changes density data");
    std::cout << (failures ? "FAIL" : "PASS") << ": bounded density operation contract\n";
    return failures ? 1 : 0;
}
