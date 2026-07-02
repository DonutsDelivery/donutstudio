// RifeEngine implementation — see rife.h. Compiled only with ARBIT_HAVE_ONNX.

#include "rife.h"

#include <onnxruntime_cxx_api.h>

extern "C"
{
#include <libavformat/avio.h>
#include <libavutil/error.h>
}

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace arbitrife
{
namespace
{

// Model: RIFE v4.18, "v2" ONNX graph (dynamic HxW, internal mod-32 padding,
// input [1,7,H,W] = frame0 RGB + frame1 RGB + timestep plane, output
// [1,3,H,W]). Weights: hzwer/Practical-RIFE (MIT), ONNX conversion published
// by the vs-mlrt project. Provenance + license recorded in PROTOCOL.md §RIFE.
constexpr const char* kModelFile = "rife_v4.18.onnx";
constexpr const char* kModelUrl =
    "https://donutsdelivery.online/download-arbit/files/video-helper/models/rife_v4.18.onnx";
constexpr const char* kModelSha256 =
    "097a83ef9024d7a340a5a987f130870f80d9e056633537e08694ca3204c0c7a8";
constexpr int64_t kModelBytes = 21532608;

// ------------------------------------------------------------------ sha256
// Compact SHA-256 (FIPS 180-4), enough to verify the downloaded model.
struct Sha256
{
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                      0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t buf[64];
    uint64_t total = 0;
    size_t fill = 0;

    static uint32_t rotr (uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void block (const uint8_t* p)
    {
        static const uint32_t K[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,
            0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
            0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,
            0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,
            0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
            0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,
            0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,
            0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
            0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2 };
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t) p[i * 4] << 24 | (uint32_t) p[i * 4 + 1] << 16
                 | (uint32_t) p[i * 4 + 2] << 8 | (uint32_t) p[i * 4 + 3];
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 = rotr (w[i - 15], 7) ^ rotr (w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr (w[i - 2], 17) ^ rotr (w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3],
                 e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t S1 = rotr (e, 6) ^ rotr (e, 11) ^ rotr (e, 25);
            const uint32_t ch = (e & f) ^ (~e & g);
            const uint32_t t1 = hh + S1 + ch + K[i] + w[i];
            const uint32_t S0 = rotr (a, 2) ^ rotr (a, 13) ^ rotr (a, 22);
            const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t t2 = S0 + mj;
            hh = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d;
        h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update (const uint8_t* p, size_t n)
    {
        total += n;
        while (n > 0)
        {
            const size_t take = std::min (n, sizeof (buf) - fill);
            std::memcpy (buf + fill, p, take);
            fill += take; p += take; n -= take;
            if (fill == sizeof (buf)) { block (buf); fill = 0; }
        }
    }

    std::string finishHex()
    {
        const uint64_t bits = total * 8;
        const uint8_t pad = 0x80;
        update (&pad, 1);
        const uint8_t zero = 0;
        while (fill != 56) update (&zero, 1);
        uint8_t len[8];
        for (int i = 0; i < 8; ++i) len[i] = (uint8_t) (bits >> (56 - i * 8));
        update (len, 8);
        char out[65];
        for (int i = 0; i < 8; ++i)
            std::snprintf (out + i * 8, 9, "%08x", h[i]);
        return std::string (out, 64);
    }
};

std::string sha256File (const std::filesystem::path& path)
{
    std::ifstream in (path, std::ios::binary);
    if (! in) return "";
    Sha256 s;
    std::vector<uint8_t> buf (1 << 16);
    while (in)
    {
        in.read ((char*) buf.data(), (std::streamsize) buf.size());
        if (in.gcount() > 0) s.update (buf.data(), (size_t) in.gcount());
    }
    return s.finishHex();
}

// ------------------------------------------------------- model acquisition

std::filesystem::path modelCacheDir()
{
    namespace fs = std::filesystem;
#if defined(_WIN32)
    if (const char* base = std::getenv ("LOCALAPPDATA"); base && *base)
        return fs::path (base) / "Arbit" / "video-helper" / "models";
    return fs::temp_directory_path() / "arbit-video-helper" / "models";
#elif defined(__APPLE__)
    if (const char* home = std::getenv ("HOME"); home && *home)
        return fs::path (home) / "Library" / "Caches" / "Arbit" / "video-helper" / "models";
    return fs::temp_directory_path() / "arbit-video-helper" / "models";
#else
    if (const char* xdg = std::getenv ("XDG_CACHE_HOME"); xdg && *xdg)
        return fs::path (xdg) / "Arbit" / "video-helper" / "models";
    if (const char* home = std::getenv ("HOME"); home && *home)
        return fs::path (home) / ".cache" / "Arbit" / "video-helper" / "models";
    return fs::temp_directory_path() / "arbit-video-helper" / "models";
#endif
}

// Download over the libavformat https client (no extra network dependency;
// requires an FFmpeg built with TLS — true for system/distro builds).
std::string downloadFile (const char* url, const std::filesystem::path& dest)
{
    AVIOContext* io = nullptr;
    const int rc = avio_open2 (&io, url, AVIO_FLAG_READ, nullptr, nullptr);
    if (rc < 0)
    {
        char ebuf[128] = {};
        av_strerror (rc, ebuf, sizeof (ebuf));
        return std::string ("model download failed (") + ebuf + "): " + url;
    }
    std::ofstream out (dest, std::ios::binary | std::ios::trunc);
    if (! out)
    {
        avio_closep (&io);
        return "cannot write " + dest.string();
    }
    std::vector<uint8_t> buf (1 << 16);
    int64_t total = 0;
    std::string error;
    for (;;)
    {
        const int n = avio_read (io, buf.data(), (int) buf.size());
        if (n == AVERROR_EOF) break;
        if (n < 0) { error = "model download interrupted"; break; }
        out.write ((const char*) buf.data(), n);
        total += n;
        if (total > kModelBytes * 2) { error = "model download larger than expected"; break; }
    }
    avio_closep (&io);
    out.close();
    if (error.empty() && ! out) error = "model write failed: " + dest.string();
    return error;
}

// Returns "" and sets pathOut on success. Verifies the pinned SHA-256 both
// for cached and freshly downloaded copies; a corrupt cache is re-fetched.
std::string ensureModel (std::filesystem::path& pathOut)
{
    namespace fs = std::filesystem;

    if (const char* override_ = std::getenv ("ARBIT_RIFE_MODEL"); override_ && *override_)
    {
        pathOut = override_;                       // test/dev escape hatch
        return fs::exists (pathOut) ? "" : "ARBIT_RIFE_MODEL not found: " + pathOut.string();
    }

    const fs::path dir = modelCacheDir();
    std::error_code ec;
    fs::create_directories (dir, ec);
    pathOut = dir / kModelFile;

    if (fs::exists (pathOut, ec) && sha256File (pathOut) == kModelSha256)
        return "";

    std::fprintf (stderr, "[rife] downloading model (%lld bytes): %s\n",
                  (long long) kModelBytes, kModelUrl);
    const fs::path tmp = dir / (std::string (kModelFile) + ".part");
    if (auto err = downloadFile (kModelUrl, tmp); ! err.empty())
    {
        fs::remove (tmp, ec);
        return err;
    }
    if (sha256File (tmp) != kModelSha256)
    {
        fs::remove (tmp, ec);
        return "model sha256 mismatch after download (corrupt or tampered)";
    }
    fs::rename (tmp, pathOut, ec);
    if (ec) return "cannot move model into cache: " + ec.message();
    return "";
}

} // namespace

// ------------------------------------------------------------------ engine

struct RifeEngine::Impl
{
    Ort::Env env { ORT_LOGGING_LEVEL_ERROR, "arbit-rife" };
    Ort::Session session { nullptr };
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu (OrtArenaAllocator, OrtMemTypeDefault);
    std::string inputName, outputName;
    std::vector<float> input;
};

RifeEngine::RifeEngine() = default;
RifeEngine::~RifeEngine() = default;

std::string RifeEngine::init()
{
    std::filesystem::path model;
    if (auto err = ensureModel (model); ! err.empty())
        return err;

    impl_ = std::make_unique<Impl>();
    // EP selector: "coreml" (Apple Neural Engine / Metal GPU), "cuda" (NVIDIA),
    // or "cpu" (no realtime EP). The accelerated EP is platform-specific; CPU is
    // always the fallback and is export-only (too slow for the realtime viewport).
    enum class Ep { CoreML, Cuda, Cpu };
    auto createSession = [&] (Ep ep) -> std::string
    {
        try
        {
            Ort::SessionOptions so;
            so.SetGraphOptimizationLevel (GraphOptimizationLevel::ORT_ENABLE_ALL);
            if (ep == Ep::Cuda)
            {
                OrtCUDAProviderOptions cu {}; // device 0, defaults
                so.AppendExecutionProvider_CUDA (cu);
            }
            else if (ep == Ep::CoreML)
            {
#if defined(__APPLE__)
                // String-keyed EP append (ORT >= 1.16): run on the Neural Engine
                // and GPU. The model is a small RIFE v4.x conv net — a good fit
                // for the ANE; falls back internally to CPU for unsupported ops.
                so.AppendExecutionProvider ("CoreML", { { "MLComputeUnits", "ALL" } });
#else
                return "CoreML EP not available on this platform";
#endif
            }
            impl_->session = Ort::Session (impl_->env, model.c_str(), so);
            return "";
        }
        catch (const Ort::Exception& e) { return e.what(); }
        catch (const std::exception& e) { return e.what(); }
    };

#if defined(__APPLE__)
    std::string accelErr = createSession (Ep::CoreML);
    const char* accelName = "rife-coreml";
#else
    const char* noCuda = std::getenv ("ARBIT_RIFE_DISABLE_CUDA"); // diagnostics
    std::string accelErr = (noCuda && *noCuda) ? "disabled by ARBIT_RIFE_DISABLE_CUDA"
                                               : createSession (Ep::Cuda);
    const char* accelName = "rife-cuda";
#endif
    if (accelErr.empty())
        backend_ = accelName;
    else
    {
        std::fprintf (stderr, "[rife] %s EP unavailable, falling back to CPU: %s\n",
                      accelName, accelErr.c_str());
        if (auto cpuErr = createSession (Ep::Cpu); ! cpuErr.empty())
        {
            impl_.reset();
            return std::string ("ONNX session creation failed (") + accelName + ": "
                 + accelErr + "; CPU: " + cpuErr + ")";
        }
        backend_ = "rife-cpu";
    }

    try
    {
        Ort::AllocatorWithDefaultOptions alloc;
        impl_->inputName = impl_->session.GetInputNameAllocated (0, alloc).get();
        impl_->outputName = impl_->session.GetOutputNameAllocated (0, alloc).get();
    }
    catch (const Ort::Exception& e)
    {
        impl_.reset();
        return std::string ("ONNX model introspection failed: ") + e.what();
    }
    std::fprintf (stderr, "[rife] ready: %s (model %s)\n", backend_.c_str(), kModelFile);
    return "";
}

std::string RifeEngine::interpolate (const uint8_t* rgba0, int stride0,
                                     const uint8_t* rgba1, int stride1,
                                     int width, int height, float timestep,
                                     std::vector<uint8_t>& rgbaOut)
{
    if (impl_ == nullptr) return "rife engine not initialized";
    if (rgba0 == nullptr || rgba1 == nullptr || width <= 0 || height <= 0)
        return "rife: bad input frame";

    const size_t plane = (size_t) width * (size_t) height;
    auto& in = impl_->input;
    in.resize (plane * 7);

    constexpr float k = 1.0f / 255.0f;
    for (int y = 0; y < height; ++y)
    {
        const uint8_t* r0 = rgba0 + (size_t) y * (size_t) stride0;
        const uint8_t* r1 = rgba1 + (size_t) y * (size_t) stride1;
        const size_t row = (size_t) y * (size_t) width;
        for (int x = 0; x < width; ++x)
        {
            in[0 * plane + row + x] = (float) r0[x * 4 + 0] * k;
            in[1 * plane + row + x] = (float) r0[x * 4 + 1] * k;
            in[2 * plane + row + x] = (float) r0[x * 4 + 2] * k;
            in[3 * plane + row + x] = (float) r1[x * 4 + 0] * k;
            in[4 * plane + row + x] = (float) r1[x * 4 + 1] * k;
            in[5 * plane + row + x] = (float) r1[x * 4 + 2] * k;
        }
    }
    std::fill (in.begin() + (ptrdiff_t) (6 * plane), in.end(),
               std::clamp (timestep, 0.0f, 1.0f));

    try
    {
        const auto t0 = std::chrono::steady_clock::now();
        const int64_t shape[4] = { 1, 7, height, width };
        Ort::Value inTensor = Ort::Value::CreateTensor<float> (
            impl_->memInfo, in.data(), in.size(), shape, 4);
        const char* inNames[] = { impl_->inputName.c_str() };
        const char* outNames[] = { impl_->outputName.c_str() };
        auto outs = impl_->session.Run (Ort::RunOptions { nullptr },
                                        inNames, &inTensor, 1, outNames, 1);
        const auto info = outs[0].GetTensorTypeAndShapeInfo();
        const auto outShape = info.GetShape();
        if (outShape.size() != 4 || outShape[1] != 3
            || outShape[2] != height || outShape[3] != width)
            return "rife: unexpected output shape";

        const float* o = outs[0].GetTensorData<float>();
        rgbaOut.resize (plane * 4);
        for (size_t i = 0; i < plane; ++i)
        {
            const auto to8 = [] (float v) -> uint8_t
            { return (uint8_t) std::clamp ((int) (v * 255.0f + 0.5f), 0, 255); };
            rgbaOut[i * 4 + 0] = to8 (o[0 * plane + i]);
            rgbaOut[i * 4 + 1] = to8 (o[1 * plane + i]);
            rgbaOut[i * 4 + 2] = to8 (o[2 * plane + i]);
            rgbaOut[i * 4 + 3] = 255;
        }
        inferMsTotal_ += std::chrono::duration<double, std::milli> (
                             std::chrono::steady_clock::now() - t0).count();
        ++inferCount_;
    }
    catch (const Ort::Exception& e)
    {
        return std::string ("rife inference failed: ") + e.what();
    }
    return "";
}

} // namespace arbitrife
