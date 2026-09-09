#include "../src/flat_shader_bridge.h"

#include <cstdio>
#include <map>
#include <memory>
#include <vector>

int main()
{
    videowire::ShaderOperationPlan mutablePlan;
    mutablePlan.revision = 9;
    mutablePlan.operations.resize(3);
    for (std::size_t index = 0; index < mutablePlan.operations.size(); ++index)
    {
        auto& operation = mutablePlan.operations[index];
        operation.nodeId = static_cast<int>(index + 11);
        operation.generatedParameters = { { "gain", static_cast<double>(index + 1) } };
    }
    mutablePlan.digest = videowire::shaderOperationPlanDigest(mutablePlan, mutablePlan.revision);
    const videowire::ImmutableShaderOperationPlan admitted =
        std::make_shared<const videowire::ShaderOperationPlan>(mutablePlan);

    const std::map<int, std::map<std::string, double>> evaluated {
        { 12, { { "gain", 42.0 } } }
    };
    std::vector<int> visits;
    std::vector<double> gains;
    const bool completed = videowire::visitOrderedShaderOperationsOnce(
        admitted, evaluated, [&](const auto& operation, const auto& parameters)
        {
            visits.push_back(operation.nodeId);
            gains.push_back(parameters.at("gain"));
            return true;
        });

    const bool onceAndOrdered = completed
        && visits == std::vector<int> { 11, 12, 13 }
        && gains == std::vector<double> { 1.0, 42.0, 3.0 };
    const bool backendParity =
        &videowire::shaderOperationParameters(admitted->operations[0], evaluated)
            == &admitted->operations[0].generatedParameters
        && videowire::shaderOperationParameters(admitted->operations[1], evaluated).at("gain") == 42.0;
    if (!onceAndOrdered || !backendParity)
    {
        std::fprintf(stderr, "shader operation plan execution/fallback contract failed\n");
        return 1;
    }
    std::printf("shader-operation-plan: ordered once with shared fallback parity\n");
    return 0;
}