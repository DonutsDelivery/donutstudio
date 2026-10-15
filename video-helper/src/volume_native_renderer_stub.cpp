#include "volume_native_renderer.h"

namespace videohelper::volume
{
NativeVolumeExecutionBackend& nativeVolumeExecutionBackend()
{
    return unavailableNativeVolumeExecutionBackend();
}
} // namespace videohelper::volume
