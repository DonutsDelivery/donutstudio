#include "../../shared/RenderPassOutputContract.h"

#include <cstdlib>
#include <iostream>
#include <limits>
#include <string_view>
#include <type_traits>
#include <utility>

using namespace renderpassoutput;

namespace
{
[[noreturn]] void fail(const char* message)
{
    std::cerr << "FAIL: " << message << '\n';
    std::exit(1);
}

void require(bool condition, const char* message)
{
    if (!condition)
        fail(message);
}

AttachmentDescription attachment(Output output, Extent extent)
{
    const auto exact = requirements(output);
    return { output, exact.format, exact.colorSpace, extent };
}

void requireRejected(const Description& description,
                     AdmissionFailure expected,
                     const char* message,
                     const AdmissionLimits& limits = {})
{
    AdmissionFailure failure = AdmissionFailure::None;
    const auto admitted = admit(description, failure, limits);
    require(!admitted.has_value() && failure == expected, message);
}
} // namespace

int main()
{
    static_assert(!std::is_default_constructible_v<AdmittedOutputs>);
    static_assert(!std::is_copy_assignable_v<AdmittedOutputs>);
    static_assert(!std::is_move_assignable_v<AdmittedOutputs>);
    static_assert(std::is_const_v<std::remove_reference_t<decltype(
        std::declval<const AdmittedOutputs&>().attachments())>>);

    require(token(Output::Color) == "color"
                && token(Output::Depth) == "depth"
                && token(Output::Normal) == "normal"
                && token(Output::Motion) == "motion"
                && token(Output::Emission) == "emission"
                && token(Output::Mask) == "mask"
                && token(Output::MaterialId) == "materialId"
                && token(Output::ObjectId) == "objectId",
            "stable output tokens changed");
    for (std::size_t index = 0; index < kOutputs.size(); ++index)
    {
        require(static_cast<std::size_t>(kOutputs[index]) == index,
                "stable output ordinal changed");
        require(outputFromToken(token(kOutputs[index])) == kOutputs[index],
                "output token did not round-trip");
    }
    require(!outputFromToken("beauty").has_value(), "unknown output token was accepted");

    const Extent extent { 64, 32 };
    Description all;
    all.extent = extent;
    for (auto found = kOutputs.rbegin(); found != kOutputs.rend(); ++found)
        all.attachments.push_back(attachment(*found, extent));

    AdmissionFailure failure = AdmissionFailure::ByteLimitExceeded;
    auto admitted = admit(all, failure);
    require(admitted.has_value() && failure == AdmissionFailure::None,
            "canonical attachment set was rejected");
    require(admitted->version() == kWireVersion && admitted->extent() == extent,
            "admission lost the exact version or extent");
    require(admitted->attachments().size() == kMaximumAttachments,
            "admission lost an attachment");
    require(admitted->totalByteCount() == 64ull * 32ull * 41ull,
            "total attachment byte accounting is wrong");

    for (std::size_t index = 0; index < admitted->attachments().size(); ++index)
    {
        const auto output = kOutputs[index];
        const auto exact = requirements(output);
        const auto& value = admitted->attachments()[index];
        require(value.output == output, "admitted attachments are not in stable ordinal order");
        require(value.format == exact.format && value.colorSpace == exact.colorSpace,
                "admission changed an exact output descriptor");
        require(value.extent == extent, "admission changed an attachment extent");
        require(value.byteCount == 64ull * 32ull * bytesPerPixel(exact.format),
                "per-attachment byte accounting is wrong");
        require(admitted->find(output) == &value, "output lookup returned the wrong attachment");
    }

    Description empty;
    empty.extent = { 1, 1 };
    auto admittedEmpty = admit(empty, failure);
    require(admittedEmpty.has_value() && admittedEmpty->attachments().empty()
                && admittedEmpty->totalByteCount() == 0,
            "a pass with no optional outputs was rejected");

    auto invalid = all;
    invalid.version = kWireVersion + 1;
    requireRejected(invalid, AdmissionFailure::UnsupportedVersion,
                    "unknown wire version was accepted");

    invalid = all;
    invalid.extent = { 0, 32 };
    requireRejected(invalid, AdmissionFailure::InvalidExtent,
                    "zero pass extent was accepted");

    invalid = all;
    invalid.extent = { kMaximumWidth, kMaximumHeight };
    for (auto& value : invalid.attachments)
        value.extent = invalid.extent;
    requireRejected(invalid, AdmissionFailure::InvalidExtent,
                    "pass extent above the hard pixel bound was accepted");

    invalid = all;
    invalid.attachments.front().extent.width += 1;
    requireRejected(invalid, AdmissionFailure::AttachmentExtentMismatch,
                    "attachment extent different from the pass was accepted");

    invalid = all;
    invalid.attachments.push_back(attachment(Output::Color, extent));
    requireRejected(invalid, AdmissionFailure::AttachmentCountExceeded,
                    "attachment count above the hard bound was accepted");

    invalid = empty;
    invalid.attachments.push_back({ static_cast<Output>(200), PixelFormat::R8Unorm,
                                    ColorSpace::Data, empty.extent });
    requireRejected(invalid, AdmissionFailure::UnknownOutput,
                    "unknown output ordinal was accepted");

    invalid = empty;
    invalid.attachments.push_back({ Output::Mask, static_cast<PixelFormat>(200),
                                    ColorSpace::Data, empty.extent });
    requireRejected(invalid, AdmissionFailure::UnknownPixelFormat,
                    "unknown pixel format was accepted");

    invalid = empty;
    invalid.attachments.push_back({ Output::Mask, PixelFormat::R8Unorm,
                                    static_cast<ColorSpace>(200), empty.extent });
    requireRejected(invalid, AdmissionFailure::UnknownColorSpace,
                    "unknown color space was accepted");

    invalid = empty;
    invalid.attachments = { attachment(Output::Color, empty.extent),
                            attachment(Output::Color, empty.extent) };
    requireRejected(invalid, AdmissionFailure::DuplicateOutput,
                    "duplicate output was accepted");

    invalid.attachments[1].format = PixelFormat::R32Float;
    requireRejected(invalid, AdmissionFailure::DuplicateOutput,
                    "incompatible duplicate output was not rejected as a duplicate");

    invalid = empty;
    invalid.attachments = { attachment(Output::Depth, empty.extent) };
    invalid.attachments[0].format = PixelFormat::RGBA16Float;
    requireRejected(invalid, AdmissionFailure::FormatMismatch,
                    "non-canonical output format was accepted");

    invalid = empty;
    invalid.attachments = { attachment(Output::Color, empty.extent) };
    invalid.attachments[0].colorSpace = ColorSpace::Data;
    requireRejected(invalid, AdmissionFailure::ColorSpaceMismatch,
                    "non-canonical output color space was accepted");

    AdmissionLimits tight;
    tight.totalBytes = admitted->totalByteCount() - 1;
    requireRejected(all, AdmissionFailure::ByteLimitExceeded,
                    "attachment set above the byte budget was accepted", tight);

    tight = {};
    tight.attachments = kMaximumAttachments - 1;
    requireRejected(all, AdmissionFailure::AttachmentCountExceeded,
                    "attachment set above the caller count budget was accepted", tight);

    tight = {};
    tight.width = extent.width - 1;
    requireRejected(all, AdmissionFailure::InvalidExtent,
                    "pass above the caller extent budget was accepted", tight);

    // Totals above 2 GiB are admitted; only device allocation stops a frame.
    Description large;
    large.extent = { 8192, 8192 };
    for (const auto output : kOutputs)
        large.attachments.push_back(attachment(output, large.extent));
    const auto wide = admit(large, failure);
    require(wide.has_value() && failure == AdmissionFailure::None
                && wide->totalByteCount() > 2ull * 1024ull * 1024ull * 1024ull,
            "a representable attachment total above 2 GiB was refused");
    tight = {};
    tight.totalBytes = std::numeric_limits<std::uint64_t>::max();
    require(admit(all, failure, tight).has_value(), "a representable byte limit was refused");

    std::uint64_t product = 0;
    require(!detail::checkedMultiply(std::numeric_limits<std::uint64_t>::max(), 2, product),
            "byte multiplication overflow was not detected");
    require(!detail::checkedAdd(std::numeric_limits<std::uint64_t>::max(), 1, product),
            "byte addition overflow was not detected");

    std::cout << "RenderPassOutputContractTests: PASS\n";
    return 0;
}
