#pragma once
//
// Shader-language front door (visual-engine plan P3).
//
// One entry point — lowerToGlsl(source, lang) — that turns any accepted shading
// language into the GLSL the renderer already runs, so the rest of the helper
// stays GLSL-only. GLSL passes through untouched; Slang and SPIR-V route through
// their front-ends (slang_frontend.h / spirv_ingest.h). The lowered GLSL then
// flows through the existing dialect front door (wrapToContract) exactly like a
// hand-written bare-GLSL shader, so no renderer change is needed.
//
// Compilation can be slow, so results are cached on disk keyed by (lang,
// FNV-1a(source)) next to the other helper caches (~/.cache/Arbit/video-helper).
//
#include "spirv_ingest.h"
#include "slang_frontend.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace arbitshadercompile
{

enum class Lang { Glsl, Slang, Spirv };

inline Lang langFromWire (const std::string& s)
{
    if (s == "slang") return Lang::Slang;
    if (s == "spirv" || s == "spir-v" || s == "spv") return Lang::Spirv;
    return Lang::Glsl;   // "", "glsl", anything else
}

inline const char* langToWire (Lang l)
{
    switch (l) { case Lang::Slang: return "slang";
                 case Lang::Spirv: return "spirv";
                 default:          return "glsl"; }
}

struct LowerResult
{
    bool ok = false;
    Lang lang = Lang::Glsl;
    std::string glsl;     // the lowered GLSL (== source when lang==Glsl)
    std::string error;    // human-readable on failure
    bool cacheHit = false;
};

namespace detail
{
    // Base64 decode (SPIR-V arrives base64 over the JSON wire). Tolerant of
    // whitespace / missing padding. Returns false on an invalid alphabet char.
    inline bool base64Decode (const std::string& in, std::vector<uint8_t>& out)
    {
        auto val = [] (char c) -> int {
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= 'a' && c <= 'z') return c - 'a' + 26;
            if (c >= '0' && c <= '9') return c - '0' + 52;
            if (c == '+') return 62;
            if (c == '/') return 63;
            return -1;
        };
        out.clear();
        int acc = 0, bits = 0;
        for (char c : in)
        {
            if (c == '=' || c == '\n' || c == '\r' || c == ' ' || c == '\t') continue;
            const int v = val (c);
            if (v < 0) return false;
            acc = (acc << 6) | v; bits += 6;
            if (bits >= 8) { bits -= 8; out.push_back ((uint8_t) ((acc >> bits) & 0xFF)); }
        }
        return true;
    }

    inline uint64_t fnv1a (const std::string& s)
    {
        uint64_t h = 1469598103934665603ull;
        for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
        return h;
    }

    inline std::filesystem::path cacheDir()
    {
        namespace fs = std::filesystem;
#if defined(_WIN32)
        if (const char* base = std::getenv ("LOCALAPPDATA"); base && *base)
            return fs::path (base) / "Arbit" / "video-helper" / "shaders";
        return fs::temp_directory_path() / "arbit-video-helper" / "shaders";
#elif defined(__APPLE__)
        if (const char* home = std::getenv ("HOME"); home && *home)
            return fs::path (home) / "Library" / "Caches" / "Arbit" / "video-helper" / "shaders";
        return fs::temp_directory_path() / "arbit-video-helper" / "shaders";
#else
        if (const char* xdg = std::getenv ("XDG_CACHE_HOME"); xdg && *xdg)
            return fs::path (xdg) / "Arbit" / "video-helper" / "shaders";
        if (const char* home = std::getenv ("HOME"); home && *home)
            return fs::path (home) / ".cache" / "Arbit" / "video-helper" / "shaders";
        return fs::temp_directory_path() / "arbit-video-helper" / "shaders";
#endif
    }

    inline std::filesystem::path cachePath (Lang lang, const std::string& source)
    {
        char name[64];
        std::snprintf (name, sizeof name, "%s-%016llx.glsl",
                       langToWire (lang), (unsigned long long) fnv1a (source));
        return cacheDir() / name;
    }
}

// Lower `source` (GLSL text, Slang text, or base64 SPIR-V) to GLSL. The GL
// version targets the renderer's 3.3 core baseline (portable; a 4.3 context
// runs it unchanged). Caches successful lowerings on disk.
inline LowerResult lowerToGlsl (const std::string& source, Lang lang,
                                int glslVersion = 330)
{
    LowerResult r;
    r.lang = lang;

    if (lang == Lang::Glsl)
    {
        r.ok = true;
        r.glsl = source;     // no lowering; the dialect front door handles it
        return r;
    }

    // Disk-cache lookup (slang/spirv only — GLSL has nothing to cache).
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path cp = detail::cachePath (lang, source);
    if (fs::exists (cp, ec))
    {
        std::ifstream in (cp, std::ios::binary);
        std::string cached ((std::istreambuf_iterator<char> (in)), {});
        if (! cached.empty())
        {
            r.ok = true; r.glsl = std::move (cached); r.cacheHit = true;
            return r;
        }
    }

    bool ok = false;
    if (lang == Lang::Slang)
    {
        ok = slangToGlsl (source, r.glsl, r.error);
    }
    else // Spirv
    {
        std::vector<uint8_t> bytes;
        if (! detail::base64Decode (source, bytes) || (bytes.size() % 4) != 0)
        {
            r.error = "SPIR-V payload is not valid base64 / not 4-byte aligned";
        }
        else
        {
            std::vector<uint32_t> words (bytes.size() / 4);
            std::memcpy (words.data(), bytes.data(), bytes.size());
            ok = spirvToGlsl (words, glslVersion, r.glsl, r.error);
        }
    }

    r.ok = ok;
    if (ok)
    {
        fs::create_directories (detail::cacheDir(), ec);
        std::ofstream out (cp, std::ios::binary | std::ios::trunc);
        out.write (r.glsl.data(), (std::streamsize) r.glsl.size());
    }
    return r;
}

} // namespace arbitshadercompile
