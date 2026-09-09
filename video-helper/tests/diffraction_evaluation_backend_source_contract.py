#!/usr/bin/env python3
from pathlib import Path
import math
import struct
import sys

root = Path(sys.argv[1]).resolve()
gl = (root / "src/gpu_backend/backend_opengl.cpp").read_text(encoding="utf-8")
metal = (root / "src/gpu_backend/backend_sokol.mm").read_text(encoding="utf-8")
cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")
shared = (root.parent / "shared/DiffractiveFoilIR.h").read_text(encoding="utf-8")

shared_contract = [
    "path * (width * height) + y * width + x",
    "required > maximumEvaluations",
    "maximumEvaluations == 0",
]
for token in shared_contract:
    assert token in shared, f"shared evaluation contract lost: {token}"

for name, source in (("OpenGL", gl), ("Metal", metal)):
    assert "nativeFixtureSpatialFoilEvaluationSchedule(" in source, (
        f"{name} does not admit the shared spatial foil schedule")
    assert "EvaluationSchedule spatialSchedule" in source, (
        f"{name} does not transport an immutable schedule value")
    assert "evaluationIndex <" in source and "maximumEvaluations" in source, (
        f"{name} shader does not enforce the exact admitted ceiling")
    assert "fixture diffraction workload exceeds backend limits" in source, (
        f"{name} lost fail-closed workload rejection")

assert "uint4 diffractionEvaluationSchedule" in metal
assert "uvec4 evaluationSchedule" in gl
assert "float3x3(u.objectMatrix)" not in metal, (
    "Metal fixture shader uses a non-portable float4x4-to-float3x3 conversion")
assert "out.litBase = out.baseColor * u.ambient.xyz" in metal, (
    "Metal fixture shader lost the OpenGL-matched unevaluated-pixel fallback")
assert "compileMetalShaderStage" in metal
assert "localizedDescription.UTF8String" in metal
assert "++submittedDiffractionDrawCount" in metal
assert "? submittedDiffractionDrawCount" in metal
assert "src/gpu_backend/backend_sokol.mm" in cmake
assert "ARBIT_HAVE_METAL_BACKEND=1" in cmake
assert "arbit-diffraction-material-metal-tests" in cmake
assert '"-framework Metal"' in cmake


def raw_metal_sources(source):
    marker = 'R"metal('
    terminator = ')metal"'
    shaders = []
    cursor = 0
    while True:
        begin = source.find(marker, cursor)
        if begin < 0:
            return shaders
        begin += len(marker)
        end = source.find(terminator, begin)
        assert end >= 0, "unterminated generated MSL source"
        shaders.append(source[begin:end])
        cursor = end + len(terminator)


def tokens(source):
    result = []
    cursor = 0
    while cursor < len(source):
        character = source[cursor]
        if character.isspace():
            cursor += 1
        elif source.startswith("//", cursor):
            newline = source.find("\n", cursor + 2)
            cursor = len(source) if newline < 0 else newline + 1
        elif source.startswith("/*", cursor):
            end = source.find("*/", cursor + 2)
            assert end >= 0, "unterminated MSL comment"
            cursor = end + 2
        elif character.isalpha() or character == "_":
            end = cursor + 1
            while end < len(source) and (source[end].isalnum() or source[end] == "_"):
                end += 1
            result.append(source[cursor:end])
            cursor = end
        elif character.isdigit() or (character == "." and cursor + 1 < len(source)
                                     and source[cursor + 1].isdigit()):
            end = cursor + 1
            while end < len(source) and source[end] in "0123456789.eE+-":
                if source[end] in "+-" and source[end - 1] not in "eE":
                    break
                end += 1
            if end < len(source) and source[end] in "fF":
                end += 1
            result.append(source[cursor:end])
            cursor = end
        else:
            result.append(character)
            cursor += 1
    return result


def statements(shader_tokens):
    statement = []
    for token in shader_tokens:
        statement.append(token)
        if token == ";":
            yield statement
            statement = []


def declarations(shader_tokens):
    result = {}
    for position, statement in enumerate(statements(shader_tokens)):
        if "=" not in statement:
            continue
        equals = statement.index("=")
        if equals >= 2 and statement[equals - 2] in ("float", "float2x2"):
            result[statement[equals - 1]] = (
                statement[equals - 2], statement[equals + 1:-1], position)
    return result


def multiplication_factors(expression):
    factors = []
    factor = []
    depth = 0
    for token in expression:
        if token in "([":
            depth += 1
        elif token in ")]":
            depth -= 1
        if token == "*" and depth == 0:
            assert factor, "empty multiplication factor"
            factors.append(factor)
            factor = []
        else:
            factor.append(token)
    assert factor, "empty multiplication factor"
    factors.append(factor)
    return factors


def float32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


def validate_angle_formula(name, generated_msl):
    shader_tokens = tokens(generated_msl)
    assert "radians" not in shader_tokens, (
        f"{name} generated MSL contains the unsupported GLSL radians builtin")
    declared = declarations(shader_tokens)
    shader_statements = list(statements(shader_tokens))
    candidates = []

    for angle_name, (angle_type, angle_expression, angle_position) in declared.items():
        if angle_type != "float":
            continue
        factors = multiplication_factors(angle_expression)
        if len(factors) != 3 or any(len(factor) != 1 for factor in (factors[0], factors[2])):
            continue
        coordinate_name = factors[0][0]
        rate = factors[1]
        constant_name = factors[2][0]
        if not (len(rate) >= 3 and len(rate) % 2 == 1 and rate[-1] == "w"
                and all(token == "." for token in rate[1::2])):
            continue
        if coordinate_name not in declared or constant_name not in declared:
            continue
        coordinate_type, coordinate_expression, _ = declared[coordinate_name]
        constant_type, constant_expression, _ = declared[constant_name]
        if coordinate_type != "float" or not ({"dot", "length"} <= set(coordinate_expression)):
            continue
        if constant_type != "float" or len(constant_expression) != 1:
            continue

        expected_matrix = [
            "float2x2", "(",
            "float2", "(", "cos", "(", angle_name, ")", ",",
            "sin", "(", angle_name, ")", ")", ",",
            "float2", "(", "-", "sin", "(", angle_name, ")", ",",
            "cos", "(", angle_name, ")", ")", ")",
        ]
        rotations = [
            (rotation_name, position)
            for rotation_name, (value_type, expression, position) in declared.items()
            if value_type == "float2x2" and expression == expected_matrix
            and position > angle_position
        ]
        if len(rotations) != 1:
            continue
        rotation_name, rotation_position = rotations[0]
        rotated_coordinates = []
        for position, statement in enumerate(shader_statements):
            if position <= rotation_position or "=" not in statement:
                continue
            equals = statement.index("=")
            left = statement[:equals]
            right = statement[equals + 1:-1]
            if (len(left) >= 3 and left[-1] == "xy"
                    and all(token == "." for token in left[-2::2])
                    and right == [rotation_name, "*"] + left):
                rotated_coordinates.append(tuple(left))
        if rotated_coordinates:
            candidates.append((factors, constant_name, coordinate_name, tuple(rate)))

    assert len(candidates) == 1, (
        f"{name} must bind one canonical coordinate-rate-radian angle to both sin and cos "
        "operands of the same canonical float2x2 used to rotate that coordinate")
    factors, constant_name, coordinate_name, groove_rate = candidates[0]
    constant_token = declared[constant_name][1][0].rstrip("fF")
    conversion = float32(float(constant_token))
    assert conversion == float32(math.pi / 180.0), (
        f"{name} conversion constant is not float32(pi / 180)")

    for coordinate, rate in (
        (-1.25, -70.0), (-0.5, 40.0), (0.0, -55.0),
        (0.375, 120.0), (1.5, -30.0),
    ):
        values = {
            (coordinate_name,): float32(coordinate),
            groove_rate: float32(rate),
            (constant_name,): conversion,
        }
        result = values[tuple(factors[0])]
        for factor in factors[1:]:
            result = float32(result * values[tuple(factor)])
        reference = math.radians(coordinate * rate)
        assert math.isclose(result, reference, rel_tol=0.0, abs_tol=1.0e-6), (
            f"{name} groove angle differs at coordinate={coordinate}, rate={rate}")


def expect_rejected(label, source):
    try:
        validate_angle_formula(label, source)
    except AssertionError:
        return
    raise AssertionError(f"structural validator accepted mutation: {label}")


def validator_self_tests():
    valid = """
        constant float turn = 0.01745329251994329577f;
        float projection = flag ? dot(delta, axis) : length(delta);
        float theta = projection * uniforms.rate.w * turn;
        float2x2 basis = float2x2(
            float2(cos(theta), sin(theta)), float2(-sin(theta), cos(theta)));
        point.xy = basis * point.xy;
    """
    validate_angle_formula("renamed identifiers", valid)
    expect_rejected("unrelated sin", valid.replace("sin(theta)", "sin(other)", 1))
    expect_rejected("missing cos", valid.replace("cos(theta)", "sin(theta)", 1))
    expect_rejected("disconnected matrix", valid.replace("point.xy = basis * point.xy", "point.xy = point.xy"))
    expect_rejected("reordered angle operands", valid.replace(
        "projection * uniforms.rate.w * turn", "uniforms.rate.w * projection * turn"))
    expect_rejected("reordered matrix operands", valid.replace(
        "float2(cos(theta), sin(theta))", "float2(sin(theta), cos(theta))"))
    expect_rejected("wrong sign", valid.replace("-sin(theta)", "sin(theta)"))
    expect_rejected("wrong constant", valid.replace(
        "0.01745329251994329577f", "0.0174533f"))


validator_self_tests()
metal_shader_sources = {
    "strict diffraction": (root / "src/diffraction_material_metal.mm").read_text(
        encoding="utf-8"),
    "fixture diffraction": metal,
}
for name, source in metal_shader_sources.items():
    generated_sources = raw_metal_sources(source)
    assert generated_sources, f"{name} generated MSL was not found"
    matching_sources = [shader for shader in generated_sources
                        if "float2x2" in tokens(shader)]
    assert len(matching_sources) == 1, (
        f"{name} must contain one generated diffraction MSL source")
    validate_angle_formula(name, matching_sources[0])

print("diffraction evaluation OpenGL/Metal source parity: PASS")
