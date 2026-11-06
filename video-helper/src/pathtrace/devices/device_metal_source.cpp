// The Metal kernel's source text (Plan 3, Render engines and devices): the MSL the
// build embeds (cmake/metal_kernel_source.py) with the host's lookup tables
// written in as constant arrays. Metal has no writable program-scope data, so
// the tables the CUDA kernel reads from its launch parameters are compiled in.
// Portable C++, so every platform's tests read the source the Mac compiles.

#include "device_metal_host.h"
#include "device_metal_params.h"
#include "device_tables.h"
#include "../bsdf.h"
#include "../spectrum.h"

#include <cstddef>
#include <cstdio>
#include <string>

extern const unsigned char kArbitMetalKernel[];
extern const std::size_t kArbitMetalKernelSize;

namespace videohelper::pathtrace::device
{
namespace
{

void appendFloat (std::string& out, float value)
{
    char text[32];
    // Nine significant digits read back as the same float.
    std::snprintf (text, sizeof (text), "%.9gf", static_cast<double> (value));
    std::string literal (text);
    if (literal.find_first_of (".e") == std::string::npos)
        literal.insert (literal.size() - 1, ".0");
    out += literal;
}

void appendRows (std::string& out, const char* name, const float (*rows)[3], std::size_t count)
{
    out += "constant float " + std::string (name) + "[" + std::to_string (count) + "][3] = {\n";
    for (std::size_t i = 0; i < count; ++i)
    {
        out += "    { ";
        for (int c = 0; c < 3; ++c)
        {
            appendFloat (out, rows[i][c]);
            out += c < 2 ? ", " : " }";
        }
        out += i + 1 < count ? ",\n" : "\n";
    }
    out += "};\n";
}

void appendList (std::string& out, const char* name, const float* values, std::size_t count)
{
    out += "constant float " + std::string (name) + "[" + std::to_string (count) + "] = {\n";
    for (std::size_t i = 0; i < count; ++i)
    {
        out += i % 8 == 0 ? "    " : " ";
        appendFloat (out, values[i]);
        out += i + 1 < count ? (i % 8 == 7 ? ",\n" : ",") : "\n";
    }
    out += "};\n";
}

} // namespace

std::string metalLookupTableSource()
{
    DeviceLookupTables tables;
    exportSpectrumTables (tables);
    exportGgxTables (tables);
    std::string out;
    appendRows (out, "arbitSpectrumWeight", tables.spectrumWeight, tables.spectrumRows);
    appendRows (out, "arbitSpectrumBasis", tables.spectrumBasis, tables.spectrumRows);
    appendList (out, "arbitGgxScale", tables.ggxScale, tables.ggxEntries);
    appendList (out, "arbitGgxBias", tables.ggxBias, tables.ggxEntries);
    out += "constant ulong kArbitSpectrumRows = " + std::to_string (tables.spectrumRows) + ";\n";
    out += "constant ulong kArbitGgxEntries = " + std::to_string (tables.ggxEntries) + ";\n";
    return out;
}

std::string metalKernelSource()
{
    std::string source (reinterpret_cast<const char*> (kArbitMetalKernel), kArbitMetalKernelSize);
    const auto marker = source.find (metal::kTablesMarker);
    if (marker != std::string::npos)
        source.replace (marker, std::string (metal::kTablesMarker).size(), metalLookupTableSource());
    return source;
}

} // namespace videohelper::pathtrace::device
