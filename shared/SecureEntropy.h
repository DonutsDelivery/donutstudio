#pragma once
#include "ProgrammableRuntimeGrant.h"
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
 #include <windows.h>
 #include <bcrypt.h>
 #if defined(_MSC_VER)
  #pragma comment(lib, "bcrypt.lib")
 #endif
#elif defined(__APPLE__)
 #include <stdlib.h>
#else
 #include <cerrno>
 #include <sys/random.h>
#endif

namespace programmableruntime::secureentropy
{
using FillFunction = bool (*) (void*, size_t);

inline bool osFill (void* destination, size_t size)
{
#if defined(_WIN32)
    return BCryptGenRandom(nullptr, static_cast<PUCHAR>(destination),
                           static_cast<ULONG>(size), BCRYPT_USE_SYSTEM_PREFERRED_RNG) == 0;
#elif defined(__APPLE__)
    arc4random_buf(destination, size);
    return true;
#else
    auto* out = static_cast<uint8_t*>(destination);
    while (size != 0)
    {
        const auto count = ::getrandom(out, size, 0);
        if (count > 0) { out += count; size -= static_cast<size_t>(count); continue; }
        if (count < 0 && errno == EINTR) continue;
        return false;
    }
    return true;
#endif
}

namespace detail {
inline bool generateSessionMaterialWith (SessionSecret& secret, uint64_t& generation,
                                         FillFunction fill)
{
    secret.fill(0);
    generation = 0;
    if (fill == nullptr || ! fill(secret.data(), secret.size())) return false;
    std::array<uint8_t, 8> generationBytes {};
    if (! fill(generationBytes.data(), generationBytes.size())) { secret.fill(0); return false; }
    for (const auto byte : generationBytes) generation = (generation << 8) | byte;
    if (generation == 0) { secret.fill(0); return false; }
    return true;
}
}
inline bool generateSessionMaterial (SessionSecret& secret, uint64_t& generation)
{
    return detail::generateSessionMaterialWith(secret, generation, osFill);
}
#if defined(PROGRAMMABLE_RUNTIME_ENTROPY_TESTING)
inline bool generateSessionMaterialForTest (SessionSecret& secret, uint64_t& generation,
                                            FillFunction fill)
{
    return detail::generateSessionMaterialWith(secret, generation, fill);
}
#endif
} // namespace programmableruntime::secureentropy
