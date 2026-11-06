#pragma once

// Lookup tables the shared code computes once on the host. A device cannot run
// those start-up computations from a function-local static, so it reads the same
// values from buffers the host uploads. exportSpectrumTables (spectrum.h) and
// exportGgxTables (bsdf.h) fill this struct on the host.

#include "device_annotations.h"

#include <cstddef>

namespace videohelper::pathtrace::device
{

// The Metal kernel's tables are constant arrays in its source (device_metal.metal),
// so the pointers name PT_TABLE_SPACE (device_annotations.h).
struct DeviceLookupTables
{
    const PT_TABLE_SPACE float (*spectrumWeight)[3] = nullptr;
    const PT_TABLE_SPACE float (*spectrumBasis)[3] = nullptr;
    std::size_t spectrumRows = 0;
    const PT_TABLE_SPACE float* ggxScale = nullptr;
    const PT_TABLE_SPACE float* ggxBias = nullptr;
    std::size_t ggxEntries = 0;
};

#if defined(__CUDACC__)
#if ! defined(PATHTRACE_DEVICE_METAL)
// Defined by the device driver TU, over its launch parameters. The Metal kernel
// defines its own, returning the tables by value.
PT_SHARED_ENTRY const DeviceLookupTables& deviceLookupTables() noexcept;
#endif
#endif

} // namespace videohelper::pathtrace::device
