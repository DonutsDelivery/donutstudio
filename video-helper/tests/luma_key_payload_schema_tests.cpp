// luma_key_payload_schema_tests.cpp — unit tests for the executable LumaKey payload
// admission (luma_key_payload_schema.h).
//
// Zero test-framework dependency (a tiny CHECK harness, like param_grammar_tests.cpp).
// Returns the failure count as the process exit code.

#include "../src/luma_key_payload_schema.h"

#include <cmath>
#include <cstdio>
#include <string>

static int g_fail = 0;
static int g_total = 0;

static void check (bool cond, const std::string& what)
{
    ++g_total;
    if (! cond)
    {
        ++g_fail;
        std::printf ("FAIL: %s\n", what.c_str());
    }
}

int main()
{
    using namespace videowire;

    // A valid payload is admitted and its four values reach the executable struct.
    {
        const std::string xml = R"(<LumaKey schemaVersion="1" low="0.2" high="0.8" softness="0.05" invert="1"/>)";
        ExecutableLumaKeyPayload payload;
        LumaKeyPayloadFailure failure = LumaKeyPayloadFailure::invalidParameter;
        check (parseExecutableLumaKeyPayload (xml, payload, &failure), "valid payload is admitted");
        check (failure == LumaKeyPayloadFailure::none, "valid payload reports no failure");
        check (std::fabs (payload.low - 0.2f) < 1e-6f, "valid payload low is 0.2");
        check (std::fabs (payload.high - 0.8f) < 1e-6f, "valid payload high is 0.8");
        check (std::fabs (payload.softness - 0.05f) < 1e-6f, "valid payload softness is 0.05");
        check (std::fabs (payload.invert - 1.0f) < 1e-6f, "valid payload invert is 1");
    }

    // low > high is refused as an invalid range, not clamped.
    {
        const std::string xml = R"(<LumaKey schemaVersion="1" low="0.8" high="0.2" softness="0.1" invert="0"/>)";
        ExecutableLumaKeyPayload payload;
        LumaKeyPayloadFailure failure = LumaKeyPayloadFailure::none;
        check (! parseExecutableLumaKeyPayload (xml, payload, &failure), "low above high is refused");
        check (failure == LumaKeyPayloadFailure::invalidRange, "low above high reports invalidRange");
    }

    // A schema version other than 1 is an unsupported schema, not a malformed payload.
    {
        const std::string xml = R"(<LumaKey schemaVersion="2" low="0" high="1" softness="0.1" invert="0"/>)";
        ExecutableLumaKeyPayload payload;
        LumaKeyPayloadFailure failure = LumaKeyPayloadFailure::none;
        check (! parseExecutableLumaKeyPayload (xml, payload, &failure), "schemaVersion 2 is refused");
        check (failure == LumaKeyPayloadFailure::unsupportedSchema,
               "schemaVersion 2 reports unsupportedSchema");
    }

    // A missing field is a malformed payload.
    {
        const std::string xml = R"(<LumaKey schemaVersion="1" low="0" high="1" invert="0"/>)";
        ExecutableLumaKeyPayload payload;
        LumaKeyPayloadFailure failure = LumaKeyPayloadFailure::none;
        check (! parseExecutableLumaKeyPayload (xml, payload, &failure), "missing softness is refused");
        check (failure == LumaKeyPayloadFailure::malformedPayload,
               "missing softness reports malformedPayload");
    }

    // Values outside the pass bounds are refused as invalid parameters, never clamped.
    {
        const std::string soft = R"(<LumaKey schemaVersion="1" low="0" high="1" softness="1.5" invert="0"/>)";
        const std::string invert = R"(<LumaKey schemaVersion="1" low="0" high="1" softness="0.1" invert="2"/>)";
        ExecutableLumaKeyPayload payload;
        LumaKeyPayloadFailure failure = LumaKeyPayloadFailure::none;
        check (! parseExecutableLumaKeyPayload (soft, payload, &failure), "softness 1.5 is refused");
        check (failure == LumaKeyPayloadFailure::invalidParameter, "softness 1.5 reports invalidParameter");
        failure = LumaKeyPayloadFailure::none;
        check (! parseExecutableLumaKeyPayload (invert, payload, &failure), "invert 2 is refused");
        check (failure == LumaKeyPayloadFailure::invalidParameter, "invert 2 reports invalidParameter");
    }

    std::printf ("luma-key-payload-schema: %d/%d checks passed\n", g_total - g_fail, g_total);
    return g_fail;
}
