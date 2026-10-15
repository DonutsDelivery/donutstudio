#include "lut_loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

std::string loadCubeLut (const std::string& path,
                         std::vector<float>& rgbOut, int& sizeOut)
{
    rgbOut.clear();
    sizeOut = 0;

    std::ifstream in (path);
    if (! in.is_open())
        return "cannot open LUT file: " + path;

    int size = 0;
    size_t expected = 0;
    std::string line;
    while (std::getline (in, line))
    {
        // Strip CR and leading whitespace.
        while (! line.empty() && (line.back() == '\r' || line.back() == '\n'))
            line.pop_back();
        size_t start = line.find_first_not_of (" \t");
        if (start == std::string::npos)
            continue;
        const std::string trimmed = line.substr (start);
        if (trimmed.empty() || trimmed[0] == '#')
            continue;

        if (trimmed.rfind ("TITLE", 0) == 0)
            continue;
        if (trimmed.rfind ("DOMAIN_MIN", 0) == 0 || trimmed.rfind ("DOMAIN_MAX", 0) == 0)
            continue; // parsed-and-ignored: domain assumed 0..1 (PROTOCOL.md)
        if (trimmed.rfind ("LUT_1D_SIZE", 0) == 0)
            return "1D LUTs are not supported (need LUT_3D_SIZE): " + path;
        if (trimmed.rfind ("LUT_3D_SIZE", 0) == 0)
        {
            size = std::atoi (trimmed.c_str() + 11);
            if (size < 2 || size > 129)
                return "bad LUT_3D_SIZE " + std::to_string (size)
                     + " (supported 2..129): " + path;
            expected = (size_t) size * (size_t) size * (size_t) size * 3;
            rgbOut.reserve (expected);
            continue;
        }

        // Data line: three floats. Reject anything else.
        const char* p = trimmed.c_str();
        char* end = nullptr;
        float v[3];
        bool ok = true;
        for (int i = 0; i < 3; ++i)
        {
            v[i] = std::strtof (p, &end);
            if (end == p) { ok = false; break; }
            p = end;
        }
        if (! ok)
        {
            // Unknown keyword lines (e.g. vendor extensions) are skipped only
            // when they start with a letter; malformed numbers are an error.
            if ((trimmed[0] >= 'A' && trimmed[0] <= 'Z')
                || (trimmed[0] >= 'a' && trimmed[0] <= 'z'))
                continue;
            return "malformed .cube data line: \"" + trimmed + "\"";
        }
        if (size == 0)
            return "data before LUT_3D_SIZE: " + path;
        if (rgbOut.size() + 3 > expected)
            return "too many data lines (expected "
                 + std::to_string (expected / 3) + "): " + path;
        rgbOut.push_back (v[0]);
        rgbOut.push_back (v[1]);
        rgbOut.push_back (v[2]);
    }

    if (size == 0)
        return "missing LUT_3D_SIZE: " + path;
    if (rgbOut.size() != expected)
        return "wrong data count (" + std::to_string (rgbOut.size() / 3) + " of "
             + std::to_string (expected / 3) + " entries): " + path;

    sizeOut = size;
    return {};
}
