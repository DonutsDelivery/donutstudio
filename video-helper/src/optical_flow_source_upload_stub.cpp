#include "optical_flow_source_lease.h"

namespace videoopticalflow
{
std::unique_ptr<NativeSourceUploader> makeNativeSourceUploader (
    std::uint64_t, std::uint64_t)
{
    return {};
}
void invalidateNativeSourceUploads (std::uintptr_t) noexcept {}
} // namespace videoopticalflow
