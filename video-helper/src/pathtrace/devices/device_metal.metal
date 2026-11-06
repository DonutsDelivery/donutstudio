// Metal device kernel (Plan 3, Render engines and devices): the single-source
// integrator body in MSL, tracing against a Metal primitive acceleration
// structure with Metal ray tracing.
//
// The build does not compile this file. cmake/MetalKernelSource.cmake inlines its
// quoted includes into one source, drops the C++ standard headers (the shims below
// stand in for what the shared code takes from them) and embeds the text in the
// helper. device_metal_host.mm puts the lookup tables in at kTablesMarker and
// compiles it with newLibraryWithSource, as the Metal backend compiles its shaders
// (gpu_backend/backend_sokol.mm), so a Metal compiler error reaches the user as the
// device's failure, by name.
//
// The shared code picks its device paths on __CUDACC__, as for HIP and oneAPI, so
// this source defines it and supplies the CUDA names those paths use over Metal.

#include <metal_stdlib>
#include <metal_atomic>
#include <metal_raytracing>

#define PATHTRACE_DEVICE_METAL 1
#define __CUDACC__ 1
#define __host__
#define __device__
#define __constant__ constant
#define __forceinline__ inline __attribute__ ((always_inline))
#define noexcept
// Metal has no double. The shared code's few double sums run in float here, and
// device_metal_host.mm uploads its double buffers (the environment's row cosines) as float.
#define double float
// Metal keywords the shared code uses as names.
#define half pt_half
#define vertex pt_vertex

// ---------------------------------------------------------------- std shims

namespace std
{
typedef uint uint32_t;
typedef int int32_t;
typedef ushort uint16_t;
typedef short int16_t;
typedef uchar uint8_t;
typedef char int8_t;
typedef ulong uint64_t;
typedef long int64_t;
typedef ulong size_t;
typedef long ptrdiff_t;
typedef ulong uintptr_t;

using ::metal::abs;
using ::metal::acos;
using ::metal::asin;
using ::metal::atan2;
using ::metal::clamp;
using ::metal::copysign;
using ::metal::cos;
using ::metal::exp;
using ::metal::exp2;
using ::metal::fabs;
using ::metal::floor;
using ::metal::fmod;
using ::metal::isfinite;
using ::metal::isinf;
using ::metal::isnan;
using ::metal::log;
using ::metal::max;
using ::metal::min;
using ::metal::pow;
using ::metal::sin;
using ::metal::sqrt;
using ::metal::tan;

using ::metal::fmax;
using ::metal::fmin;

// std::min<T> and std::max<T> with the type named: Metal's are not templates.
template <typename T>
inline T min (T a, T b) { return b < a ? b : a; }
template <typename T>
inline T max (T a, T b) { return a < b ? b : a; }

template <typename T>
constant constexpr bool is_trivially_copyable_v = __is_trivially_copyable (T);

// Metal has no derived classes, so the limits the shared code reads are spelled
// out per type.
template <typename T>
struct numeric_limits;
template <>
struct numeric_limits<float>
{
    static constexpr float max() { return FLT_MAX; }
    static constexpr float min() { return FLT_MIN; }
    static constexpr float lowest() { return -FLT_MAX; }
    static constexpr float epsilon() { return FLT_EPSILON; }
    static constexpr float infinity() { return INFINITY; }
    static constexpr float quiet_NaN() { return NAN; }
};
template <>
struct numeric_limits<uint>
{
    static constexpr uint max() { return UINT_MAX; }
    static constexpr uint min() { return 0u; }
    static constexpr uint lowest() { return 0u; }
};
template <>
struct numeric_limits<int>
{
    static constexpr int max() { return INT_MAX; }
    static constexpr int min() { return INT_MIN; }
    static constexpr int lowest() { return INT_MIN; }
};
template <>
struct numeric_limits<ulong>
{
    static constexpr ulong max() { return ~0ul; }
    static constexpr ulong min() { return 0ul; }
    static constexpr ulong lowest() { return 0ul; }
};
template <>
struct numeric_limits<ushort>
{
    static constexpr ushort max() { return 0xffff; }
    static constexpr ushort min() { return 0; }
    static constexpr ushort lowest() { return 0; }
};
template <>
struct numeric_limits<uchar>
{
    static constexpr uchar max() { return 0xff; }
    static constexpr uchar min() { return 0; }
    static constexpr uchar lowest() { return 0; }
};

inline float hypot (float x, float y) { return ::metal::sqrt (x * x + y * y); }
// The next float after `from` towards `to`, as C's nextafter: MSL 3.0 has none.
inline float nextafter (float from, float to)
{
    if (::metal::isnan (from) || ::metal::isnan (to))
        return from + to;
    if (from == to)
        return to;
    if (from == 0.0f)
        return as_type<float> (1u) * (to > 0.0f ? 1.0f : -1.0f);
    const uint bits = as_type<uint> (from);
    return as_type<float> ((from < to) == (from > 0.0f) ? bits + 1u : bits - 1u);
}

template <typename T, size_t N>
struct array
{
    T values[N];
    thread T& operator[] (size_t index) thread { return values[index]; }
    const thread T& operator[] (size_t index) const thread { return values[index]; }
    const constant T& operator[] (size_t index) const constant { return values[index]; }
    const device T& operator[] (size_t index) const device { return values[index]; }
    static constexpr size_t size() { return N; }
    thread T* data() thread { return values; }
    const thread T* data() const thread { return values; }
    thread T* begin() thread { return values; }
    thread T* end() thread { return values + N; }
    const thread T* begin() const thread { return values; }
    const thread T* end() const thread { return values + N; }
};
} // namespace std

// ---------------------------------------------------------------- CUDA names

inline ::metal::uint __float_as_uint (float value) { return as_type<::metal::uint> (value); }
inline float __uint_as_float (::metal::uint bits) { return as_type<float> (bits); }
using ::metal::isfinite;

inline float atomicAdd (device float* address, float value)
{
    return ::metal::atomic_fetch_add_explicit (reinterpret_cast<device ::metal::atomic_float*> (address), value,
                                               ::metal::memory_order_relaxed);
}

// Metal has no 64-bit atomic add: the low word adds and carries into the high one.
// Totals only read after the launch, so the two words need not move together.
inline ::metal::ulong atomicAdd (device ::metal::ulong* address, ::metal::ulong value)
{
    device ::metal::atomic_uint* words = reinterpret_cast<device ::metal::atomic_uint*> (address);
    const ::metal::uint low = static_cast<::metal::uint> (value);
    ::metal::uint high = static_cast<::metal::uint> (value >> 32);
    const ::metal::uint old = ::metal::atomic_fetch_add_explicit (words, low, ::metal::memory_order_relaxed);
    if (old + low < old)
        ++high;
    if (high != 0)
        ::metal::atomic_fetch_add_explicit (words + 1, high, ::metal::memory_order_relaxed);
    return old;
}

// ---------------------------------------------------------------- tables

// ARBIT_METAL_LOOKUP_TABLES

#include "device_tables.h"

namespace videohelper::pathtrace::device
{
inline DeviceLookupTables deviceLookupTables()
{
    DeviceLookupTables tables;
    tables.spectrumWeight = arbitSpectrumWeight;
    tables.spectrumBasis = arbitSpectrumBasis;
    tables.spectrumRows = kArbitSpectrumRows;
    tables.ggxScale = arbitGgxScale;
    tables.ggxBias = arbitGgxBias;
    tables.ggxEntries = kArbitGgxEntries;
    return tables;
}
} // namespace videohelper::pathtrace::device

// ---------------------------------------------------------------- the device

#include "device_interface.h"
#include "device_metal_params.h"
#include "gpu_film.h"
#include "scene_view.h"
#include "../sampler.h"

namespace videohelper::pathtrace::device
{

// The host defines ARBIT_METAL_MOTION when the GPU has primitive motion blur.
// Its structure then holds every triangle at shutter open and close, and a ray
// meets the triangles where they are at its time; a still scene gives both
// keyframes the same vertices.
#if defined(ARBIT_METAL_MOTION)
#define ARBIT_METAL_TAGS ::metal::raytracing::triangle_data, ::metal::raytracing::primitive_motion
#define ARBIT_METAL_STRUCTURE ::metal::raytracing::acceleration_structure<::metal::raytracing::primitive_motion>
#else
#define ARBIT_METAL_TAGS ::metal::raytracing::triangle_data
#define ARBIT_METAL_STRUCTURE ::metal::raytracing::primitive_acceleration_structure
#endif

// The structure the host built over scene.geometry.positions, three vertices per
// triangle, so a primitive id is the scene's triangle index.
struct MetalAccel
{
    ARBIT_METAL_STRUCTURE structure;
    ::metal::raytracing::intersection_function_table<ARBIT_METAL_TAGS> functions;
};

// The triangle a ray leaves from, which the intersection function skips.
struct SkipPayload
{
    std::uint32_t ignoreTriangle;
};

struct MetalDevice
{
    using Accel = MetalAccel;
    using Film = gpu::Film;
    using Sampler = PathSampler;

    PT_DEVICE_FN static Sampler makeSampler (std::uint32_t pixel, std::uint32_t sampleIndex, std::uint64_t generation,
                                             SamplePattern pattern)
    {
        return Sampler (pixel, sampleIndex, generation, pattern);
    }

    PT_DEVICE_FN static bool trace (thread const Accel& accel, thread const Ray& ray, bool anyHit, thread Hit& hit)
    {
        ::metal::raytracing::ray query (float3 (ray.origin.x, ray.origin.y, ray.origin.z),
                                        float3 (ray.direction.x, ray.direction.y, ray.direction.z), ray.tMin, ray.tMax);
        ::metal::raytracing::intersector<ARBIT_METAL_TAGS> intersector;
        intersector.assume_geometry_type (::metal::raytracing::geometry_type::triangle);
        intersector.accept_any_intersection (anyHit);
        SkipPayload payload { ray.ignoreTriangle };
#if defined(ARBIT_METAL_MOTION)
        const auto result = intersector.intersect (query, accel.structure, ray.time, accel.functions, payload);
#else
        const auto result = intersector.intersect (query, accel.structure, accel.functions, payload);
#endif
        if (result.type == ::metal::raytracing::intersection_type::none)
            return false;
        hit.t = result.distance;
        hit.u = result.triangle_barycentric_coord.x;
        hit.v = result.triangle_barycentric_coord.y;
        hit.triangle = result.primitive_id;
        hit.time = ray.time;
        return true;
    }

    PT_DEVICE_FN static bool intersect (thread const Accel& accel, thread const Ray& ray, thread Hit& hit)
    {
        return trace (accel, ray, false, hit);
    }
    PT_DEVICE_FN static bool occluded (thread const Accel& accel, thread const Ray& ray)
    {
        Hit hit;
        return trace (accel, ray, true, hit);
    }
    PT_DEVICE_FN static void filmAdd (thread Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) { gpu::filmAdd (film, x, y, rgb); }
    PT_DEVICE_FN static void filmSplat (thread Film& film, std::uint32_t x, std::uint32_t y, Vec3 rgb) { gpu::filmSplat (film, x, y, rgb); }
    PT_DEVICE_FN static void filmCountLightPath (thread Film& film) { gpu::filmCountLightPath (film); }
    PT_DEVICE_FN static void filmGuide (thread Film& film, std::uint32_t x, std::uint32_t y, Vec3 albedo, Vec3 normal)
    {
        gpu::filmGuide (film, x, y, albedo, normal);
    }
};

} // namespace videohelper::pathtrace::device

// One source, so the out-of-line shared code comes in as text, as for CUDA.
#include "../spectrum.cpp"
#include "../bsdf.cpp"
#include "../environment_map.cpp"
#include "../lights.cpp"

#define PT_DEVICE ::videohelper::pathtrace::device::MetalDevice
#define PT_DEVICE_NAMESPACE metalkernel
#include "integrator_body.inl"

// Accepts every triangle but the one the ray leaves from. Runs because the host
// marks the geometry non-opaque. The attribute names its tags unqualified.
using ::metal::raytracing::triangle_data;
using ::metal::raytracing::primitive_motion;
#if defined(ARBIT_METAL_MOTION)
[[intersection (triangle, triangle_data, primitive_motion)]]
#else
[[intersection (triangle, triangle_data)]]
#endif
bool arbitSkipIgnoredTriangle (::metal::uint primitive [[primitive_id]],
                               ray_data ::videohelper::pathtrace::device::SkipPayload& payload [[payload]])
{
    return primitive != payload.ignoreTriangle;
}

// One thread per pixel, the same per-sample loop as the CUDA kernel.
kernel void arbitPathTraceMetal (
    constant ::videohelper::pathtrace::device::metal::LaunchParams& params [[buffer (0)]],
    ARBIT_METAL_STRUCTURE structure [[buffer (1)]],
    ::metal::raytracing::intersection_function_table<ARBIT_METAL_TAGS> functions [[buffer (2)]],
    ::metal::uint2 pixel [[thread_position_in_grid]])
{
    namespace pt = ::videohelper::pathtrace;
    const std::uint32_t x = pixel.x;
    const std::uint32_t y = pixel.y;
    if (x >= params.film.width || y >= params.height)
        return;
    pt::device::SceneView<pt::device::MetalAccel> view;
    view.positions = params.view.positions;
    view.motionOpen = params.view.motionOpen;
    view.motionClose = params.view.motionClose;
    view.triangles = params.view.triangles;
    view.materials = params.view.materials;
    view.fallbackMaterial = params.view.fallbackMaterial;
    view.gratings = params.view.gratings;
    view.accel.structure = structure;
    view.accel.functions = functions;
    // The light set's member functions run on the kernel's own copy.
    pt::LightSetView lights;
    if (params.view.lights != nullptr)
    {
        lights = *params.view.lights;
        view.lights = &lights;
    }
    view.anyPartialCoverage = params.view.anyPartialCoverage;
    view.anyThinSheet = params.view.anyThinSheet;
    view.anyDispersive = params.view.anyDispersive;
    view.anyMediumBoundary = params.view.anyMediumBoundary;
    view.anyVolume = params.view.anyVolume;
    view.worldMedium = params.view.worldMedium;
    view.worldVolumeRadius = params.view.worldVolumeRadius;
    view.anyHiddenObjects = params.view.anyHiddenObjects;
    view.boundsCentre = params.view.boundsCentre;
    view.boundsRadius = params.view.boundsRadius;
    const pt::IntegratorSettings settings = params.settings;
    const pt::PinholeCamera camera = params.camera;
    pt::device::gpu::Film film = params.film;
    pt::PathCounters counters;
    std::uint64_t added = 0, nonFinite = 0;
    for (std::uint32_t s = 0; s < params.samples; ++s)
    {
        const auto outcome = pt::device::metalkernel::renderSample (view, settings, camera, x, y, params.film.width,
                                                                    params.firstSample + s, params.generation, film, counters);
        if (outcome == pt::device::SampleOutcome::NonFinite)
            ++nonFinite;
        else
            ++added;
    }
    pt::device::gpu::addTotals (params.totals, counters, added, nonFinite);
}
