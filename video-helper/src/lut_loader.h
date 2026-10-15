// lut_loader.h — .cube 3D LUT file parser (Adobe/Resolve cube format).
//
// Pure CPU, no GL: the viewport parses on the RPC thread and uploads on the
// render thread; the exporter parses at job start. Compiled with the
// viewport-gated sources because only the GL paths consume LUTs.
//
// Supported: LUT_3D_SIZE 2..129 (17/33/65 are the sizes in the wild),
// TITLE / comment lines, DOMAIN_MIN/DOMAIN_MAX parsed but assumed 0..1
// (non-unit domains are accepted and treated as 0..1 — documented in
// PROTOCOL.md). LUT_1D_SIZE files are rejected.
#pragma once

#include <string>
#include <vector>

// Parses a .cube file into size^3 RGB float triples, .cube data order
// (red index fastest, then green, then blue). Returns "" on success,
// an error message otherwise.
std::string loadCubeLut (const std::string& path,
                         std::vector<float>& rgbOut, int& sizeOut);
