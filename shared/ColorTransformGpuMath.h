#pragma once

#include "ColorTransformContract.h"

#include <array>

namespace colortransform::gpumath
{
using Matrix3 = std::array<double, 9>;

inline const Matrix3& rgbToXyz(ColorSpace space) noexcept
{
    static const Matrix3 srgb {
        0.4123907993, 0.3575843394, 0.1804807884,
        0.2126390059, 0.7151686788, 0.0721923154,
        0.0193308187, 0.1191947798, 0.9505321522 };
    static const Matrix3 displayP3 {
        0.4865709486, 0.2656676932, 0.1982172852,
        0.2289745641, 0.6917385218, 0.0792869141,
        0.0000000000, 0.0451133819, 1.0439443689 };
    static const Matrix3 rec2020 {
        0.6369580483, 0.1446169036, 0.1688809752,
        0.2627002120, 0.6779980715, 0.0593017165,
        0.0000000000, 0.0280726930, 1.0609850577 };
    return space == ColorSpace::DisplayP3 ? displayP3
        : space == ColorSpace::Rec2020 ? rec2020 : srgb;
}

inline const Matrix3& xyzToRgb(ColorSpace space) noexcept
{
    static const Matrix3 srgb {
         3.2409699419, -1.5373831776, -0.4986107603,
        -0.9692436363,  1.8759675015,  0.0415550574,
         0.0556300797, -0.2039769589,  1.0569715142 };
    static const Matrix3 displayP3 {
         2.4934969119, -0.9313836179, -0.4027107845,
        -0.8294889696,  1.7626640603,  0.0236246858,
         0.0358458302, -0.0761723893,  0.9568845240 };
    static const Matrix3 rec2020 {
         1.7166511880, -0.3556707838, -0.2533662814,
        -0.6666843518,  1.6164812366,  0.0157685458,
         0.0176398574, -0.0427706133,  0.9421031212 };
    return space == ColorSpace::DisplayP3 ? displayP3
        : space == ColorSpace::Rec2020 ? rec2020 : srgb;
}

inline Matrix3 multiply(const Matrix3& left, const Matrix3& right) noexcept
{
    Matrix3 result {};
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            for (int inner = 0; inner < 3; ++inner)
                result[static_cast<std::size_t>(row * 3 + column)]
                    += left[static_cast<std::size_t>(row * 3 + inner)]
                     * right[static_cast<std::size_t>(inner * 3 + column)];
    return result;
}

inline std::array<float, 16> matrix4(const Matrix3& value) noexcept
{
    std::array<float, 16> result {};
    result[15] = 1.0f;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            result[static_cast<std::size_t>(column * 4 + row)]
                = static_cast<float>(value[static_cast<std::size_t>(row * 3 + column)]);
    return result;
}

inline std::array<float, 16> inputToWorking(const Description& description) noexcept
{
    return matrix4(multiply(xyzToRgb(description.workingColorSpace),
                            rgbToXyz(description.input.colorSpace)));
}

inline std::array<float, 16> workingToOutput(const Description& description) noexcept
{
    return matrix4(multiply(xyzToRgb(description.output.colorSpace),
                            rgbToXyz(description.workingColorSpace)));
}

inline std::array<float, 4> workingLuma(const Description& description) noexcept
{
    const auto& matrix = rgbToXyz(description.workingColorSpace);
    return { static_cast<float>(matrix[3]), static_cast<float>(matrix[4]),
             static_cast<float>(matrix[5]), 0.0f };
}
} // namespace colortransform::gpumath
