#pragma once

// Lookup tables the shared code computes once on the host. A device cannot run
// those start-up computations from a function-local static, so it reads the same
// values from buffers the host uploads. exportSpectrumTables (spectrum.h) and
// exportGgxTables (bsdf.h) fill this struct on the host.

#include "device_annotations.h"

#include <cstddef>

namespace videohelper::pathtrace::device
{

struct DeviceLookupTables
{
    const float (*spectrumWeight)[3] = nullptr;
    const float (*spectrumBasis)[3] = nullptr;
    std::size_t spectrumRows = 0;
    const float* ggxScale = nullptr;
    const float* ggxBias = nullptr;
    std::size_t ggxEntries = 0;
};

#if defined(__CUDACC__)
// Defined by the device driver TU, over its launch parameters.
PT_SHARED_ENTRY const DeviceLookupTables& deviceLookupTables() noexcept;
#endif

} // namespace videohelper::pathtrace::device
