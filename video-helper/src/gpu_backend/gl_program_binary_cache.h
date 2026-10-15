#pragma once

// Persistent OpenGL program binary cache for the fixture scene program.
//
// A program linked from the fixture vertex and fragment sources can be saved
// as a driver binary. A later session on the same driver loads that binary with
// glProgramBinary and skips both shader compiles and the link. The key covers
// both sources, GL_VENDOR, GL_RENDERER, GL_VERSION and kFormatVersion, so a
// driver or shader change misses and compiles normally.
//
// Any defect in an entry (short or oversized file, wrong magic or version,
// payload digest mismatch, or a driver that rejects the binary) deletes that
// entry and falls back to the normal compile. Nothing reaches the driver
// before the structural and digest checks pass.
//
// The file logic has no GL dependency. Driver is the GL seam. The backend
// implements it over arbitgl::GlFuncs, and the unit tests implement it with a
// fake.
//
// The default location, used when ARBIT_GL_PROGRAM_CACHE_DIR is unset, depends
// on the platform.
//   Linux    $XDG_CACHE_HOME or $HOME/.cache, then DonutStudio/gl-program-cache
//   macOS    $HOME/Library/Caches, then DonutStudio/gl-program-cache
//   Windows  %LOCALAPPDATA%, then DonutStudio/gl-program-cache
// An empty ARBIT_GL_PROGRAM_CACHE_DIR disables the cache, and so does a missing
// base directory. The cache never falls back to a shared temporary directory.

#include "../sha256.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace arbitglprogramcache
{

// Bump when the entry layout or the key inputs change.
inline constexpr std::uint32_t kFormatVersion = 1;

inline constexpr char kEnvironmentVariable[] = "ARBIT_GL_PROGRAM_CACHE_DIR";

// Entry layout. Integers are little-endian.
//   offset 0    8 bytes   magic "DSGLPBIN"
//   offset 8    4 bytes   kFormatVersion
//   offset 12   4 bytes   binaryFormat (GLenum)
//   offset 16   8 bytes   payload length
//   offset 24   64 bytes  SHA-256 of the payload, lowercase hex
//   offset 88             payload
inline constexpr char kMagic[] = "DSGLPBIN";
inline constexpr std::size_t kMagicBytes = 8;
inline constexpr std::size_t kDigestHexLength = 64;
inline constexpr std::size_t kHeaderBytes = 8 + 4 + 4 + 8 + kDigestHexLength;
inline constexpr char kEntrySuffix[] = ".glpb";
inline constexpr char kTemporarySuffix[] = ".tmp";

struct Limits
{
    std::size_t maximumEntries = 64;
    std::uintmax_t maximumBytes = 256ull * 1024ull * 1024ull;
};

// Identity of the context that produced a binary. A binary loads only on the
// same vendor, renderer and version.
struct DriverIdentity
{
    std::string vendor;
    std::string renderer;
    std::string version;
};

// GL seam. Each method is one GL call or a query built from one.
class Driver
{
public:
    virtual ~Driver() = default;

    // GL_NUM_PROGRAM_BINARY_FORMATS, or zero when the entry points are missing.
    virtual int binaryFormatCount() const = 0;
    virtual DriverIdentity identity() const = 0;
    virtual std::uint32_t createProgram() = 0;
    virtual void deleteProgram (std::uint32_t program) = 0;

    // glProgramParameteri(program, GL_PROGRAM_BINARY_RETRIEVABLE_HINT, GL_TRUE).
    // Called before the program is linked.
    virtual void setRetrievableHint (std::uint32_t program) = 0;

    // glProgramBinary, then GL_LINK_STATUS. False when the driver rejects the binary.
    virtual bool loadBinary (std::uint32_t program, std::uint32_t binaryFormat,
                             const std::uint8_t* data, std::size_t size) = 0;

    // GL_PROGRAM_BINARY_LENGTH, then glGetProgramBinary. False when no binary is available.
    virtual bool retrieveBinary (std::uint32_t program, std::uint32_t& binaryFormat,
                                 std::vector<std::uint8_t>& payload) = 0;
};

// Stable key for one vertex and fragment pair on one driver. Every field is
// length-prefixed, so different inputs cannot produce the same byte stream.
inline std::string programKey (const std::string& vertexSource, const std::string& fragmentSource,
                               const DriverIdentity& identity,
                               std::uint32_t formatVersion = kFormatVersion)
{
    videohelper::Sha256 hash;
    const auto field = [&hash] (const std::string& text)
    {
        unsigned char length[8];
        const auto size = static_cast<std::uint64_t> (text.size());
        for (int i = 0; i < 8; ++i)
            length[i] = static_cast<unsigned char> (size >> (8 * i));
        hash.update (length, sizeof length);
        hash.update (text.data(), text.size());
    };
    field ("donutstudio gl program binary");
    field (std::to_string (formatVersion));
    field (identity.vendor);
    field (identity.renderer);
    field (identity.version);
    field (vertexSource);
    field (fragmentSource);
    return hash.finishHex();
}

inline std::filesystem::path entryPath (const std::filesystem::path& directory, const std::string& key)
{
    return directory / (key + kEntrySuffix);
}

// Pure resolution from environment values, so every branch can be tested
// without touching the process environment. A null pointer means unset.
inline std::optional<std::filesystem::path> resolveDirectory (const char* override_,
                                                              const char* xdgCacheHome,
                                                              const char* home,
                                                              const char* localAppData)
{
    namespace fs = std::filesystem;
    if (override_ != nullptr)
    {
        if (*override_ == '\0')
            return std::nullopt;
        return fs::path (override_);
    }
    // Each platform reads only its own base variables.
    (void) xdgCacheHome;
    (void) home;
    (void) localAppData;
    const fs::path leaf = fs::path ("DonutStudio") / "gl-program-cache";
#if defined(_WIN32)
    if (localAppData != nullptr && *localAppData != '\0')
        return fs::path (localAppData) / leaf;
    return std::nullopt;
#elif defined(__APPLE__)
    if (home != nullptr && *home != '\0')
        return fs::path (home) / "Library" / "Caches" / leaf;
    return std::nullopt;
#else
    if (xdgCacheHome != nullptr && *xdgCacheHome != '\0')
        return fs::path (xdgCacheHome) / leaf;
    if (home != nullptr && *home != '\0')
        return fs::path (home) / ".cache" / leaf;
    return std::nullopt;
#endif
}

inline std::optional<std::filesystem::path> cacheDirectory()
{
    return resolveDirectory (std::getenv (kEnvironmentVariable), std::getenv ("XDG_CACHE_HOME"),
                             std::getenv ("HOME"), std::getenv ("LOCALAPPDATA"));
}

inline void appendLittleEndian (std::vector<std::uint8_t>& bytes, std::uint64_t value, std::size_t width)
{
    for (std::size_t i = 0; i < width; ++i)
        bytes.push_back (static_cast<std::uint8_t> (value >> (8 * i)));
}

inline std::uint64_t readLittleEndian (const std::uint8_t* at, std::size_t width)
{
    std::uint64_t value = 0;
    for (std::size_t i = 0; i < width; ++i)
        value |= static_cast<std::uint64_t> (at[i]) << (8 * i);
    return value;
}

inline std::string payloadDigest (const std::uint8_t* data, std::size_t size)
{
    return videohelper::sha256Bytes (data, size);
}

inline std::vector<std::uint8_t> encodeEntry (std::uint32_t binaryFormat,
                                              const std::vector<std::uint8_t>& payload)
{
    std::vector<std::uint8_t> bytes;
    bytes.reserve (kHeaderBytes + payload.size());
    for (std::size_t i = 0; i < kMagicBytes; ++i)
        bytes.push_back (static_cast<std::uint8_t> (kMagic[i]));
    appendLittleEndian (bytes, kFormatVersion, 4);
    appendLittleEndian (bytes, binaryFormat, 4);
    appendLittleEndian (bytes, payload.size(), 8);
    const auto digest = payloadDigest (payload.data(), payload.size());
    for (const char c : digest)
        bytes.push_back (static_cast<std::uint8_t> (c));
    bytes.insert (bytes.end(), payload.begin(), payload.end());
    return bytes;
}

// Returns false for any structural or digest defect. Outputs are only
// meaningful when this returns true.
inline bool decodeEntry (const std::vector<std::uint8_t>& bytes, std::uint32_t& binaryFormat,
                         std::vector<std::uint8_t>& payload)
{
    if (bytes.size() <= kHeaderBytes)
        return false;
    if (std::memcmp (bytes.data(), kMagic, kMagicBytes) != 0)
        return false;
    if (readLittleEndian (bytes.data() + 8, 4) != kFormatVersion)
        return false;
    const auto length = readLittleEndian (bytes.data() + 16, 8);
    if (length != static_cast<std::uint64_t> (bytes.size() - kHeaderBytes))
        return false;
    const auto* body = bytes.data() + kHeaderBytes;
    const std::string digest (reinterpret_cast<const char*> (bytes.data() + 24), kDigestHexLength);
    if (digest != payloadDigest (body, static_cast<std::size_t> (length)))
        return false;
    binaryFormat = static_cast<std::uint32_t> (readLittleEndian (bytes.data() + 12, 4));
    payload.assign (body, body + static_cast<std::size_t> (length));
    return true;
}

// Reads a whole entry file. Rejects files larger than the payload limit plus
// the header, so a damaged file cannot force a large read.
inline bool readFile (const std::filesystem::path& path, std::uintmax_t maximumPayload,
                      std::vector<std::uint8_t>& bytes)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    const auto size = fs::file_size (path, ec);
    if (ec || size > maximumPayload + kHeaderBytes)
        return false;
    std::ifstream in (path, std::ios::binary);
    if (! in)
        return false;
    bytes.resize (static_cast<std::size_t> (size));
    in.read (reinterpret_cast<char*> (bytes.data()), static_cast<std::streamsize> (size));
    return static_cast<std::uintmax_t> (in.gcount()) == size;
}

inline void touchEntry (const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::last_write_time (path, std::filesystem::file_time_type::clock::now(), ec);
}

inline void removeEntry (const std::filesystem::path& path)
{
    std::error_code ec;
    std::filesystem::remove (path, ec);
}

// Removes the oldest entries until both limits hold. The entry just written is
// never removed. A temporary file older than an hour was left by a writer that
// stopped mid-write, so it is removed too.
inline void evictEntries (const std::filesystem::path& directory, const std::filesystem::path& keep,
                          const Limits& limits)
{
    namespace fs = std::filesystem;
    struct Candidate
    {
        fs::path file;
        fs::file_time_type written;
        std::uintmax_t bytes;
    };
    std::vector<Candidate> entries;
    std::uintmax_t total = 0;
    const auto now = fs::file_time_type::clock::now();
    std::error_code ec;
    for (fs::directory_iterator it (directory, ec), end; ! ec && it != end; it.increment (ec))
    {
        const fs::path file = it->path();
        std::error_code entryError;
        const auto bytes = fs::file_size (file, entryError);
        const auto written = fs::last_write_time (file, entryError);
        if (entryError)
            continue;
        if (file.extension().string() == kTemporarySuffix)
        {
            if (now - written > std::chrono::hours (1))
            {
                std::error_code removeError;
                fs::remove (file, removeError);
            }
            continue;
        }
        if (file.extension().string() != kEntrySuffix)
            continue;
        entries.push_back (Candidate {file, written, bytes});
        total += bytes;
    }
    std::sort (entries.begin(), entries.end(), [] (const Candidate& a, const Candidate& b)
    {
        return a.written < b.written;
    });
    std::size_t count = entries.size();
    for (const auto& candidate : entries)
    {
        if (count <= limits.maximumEntries && total <= limits.maximumBytes)
            break;
        if (candidate.file == keep)
            continue;
        std::error_code removeError;
        fs::remove (candidate.file, removeError);
        if (! removeError)
        {
            --count;
            total -= candidate.bytes;
        }
    }
}

// Writes one entry through a temporary file in the same directory, then
// renames it into place, and enforces the limits. Returns an empty string on
// success and the reason otherwise.
inline std::string writeEntry (const std::filesystem::path& directory, const std::string& key,
                               std::uint32_t binaryFormat, const std::vector<std::uint8_t>& payload,
                               const Limits& limits)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::create_directories (directory, ec);
    if (! fs::is_directory (directory, ec))
        return "the cache directory cannot be created";
    static std::atomic<std::uint64_t> sequence {0};
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path temporary = directory / (key + "." + std::to_string (stamp) + "."
                                            + std::to_string (sequence++) + kTemporarySuffix);
    const fs::path destination = entryPath (directory, key);
    const auto bytes = encodeEntry (binaryFormat, payload);
    {
        std::ofstream out (temporary, std::ios::binary | std::ios::trunc);
        out.write (reinterpret_cast<const char*> (bytes.data()), static_cast<std::streamsize> (bytes.size()));
        out.close();
        if (! out)
        {
            fs::remove (temporary, ec);
            return "the temporary entry cannot be written";
        }
    }
    fs::rename (temporary, destination, ec);
    if (ec)
    {
        fs::remove (temporary, ec);
        return "the entry cannot be published";
    }
    evictEntries (directory, destination, limits);
    return {};
}

// Which optional entry points the driver resolved. Reported in the state line.
struct EntryPointReport
{
    bool programBinary = false;
    bool getProgramBinary = false;
    bool programParameteri = false;
};

// One line that says whether the cache is enabled and why it is not. The
// caller passes the format count it already queried, so the driver is not asked
// again. Enabled needs a directory and at least one binary format.
inline std::string stateLine (const std::optional<std::filesystem::path>& directory,
                              int binaryFormats, const EntryPointReport& entries)
{
    if (directory.has_value() && binaryFormats > 0)
        return "gl program cache: enabled at " + directory->string()
            + " (binaryFormats=" + std::to_string (binaryFormats) + ")";
    const auto yesNo = [] (bool value) { return value ? "y" : "n"; };
    return "gl program cache: disabled (directory="
        + (directory.has_value() ? directory->string() : std::string ("none"))
        + ", binaryFormats=" + std::to_string (binaryFormats)
        + ", entryPoints=ProgramBinary " + yesNo (entries.programBinary)
        + ",GetProgramBinary " + yesNo (entries.getProgramBinary)
        + ",ProgramParameteri " + yesNo (entries.programParameteri) + ")";
}

// One handle per preparation. Cache defects never throw. They return zero or
// false, and the caller compiles normally.
class ProgramBinaryCache
{
public:
    ProgramBinaryCache (Driver& driver, std::optional<std::filesystem::path> directory,
                        Limits limits = {})
        : driver_ (driver), directory_ (std::move (directory)), limits_ (limits)
    {}

    // True when a directory is configured and the driver exposes binary formats.
    bool enabled() const
    {
        return directory_.has_value() && driver_.binaryFormatCount() > 0;
    }

    // The configured directory, or nullopt when the cache has none.
    const std::optional<std::filesystem::path>& directory() const { return directory_; }

    // Returns a linked program for this exact pair, or zero on a miss. An entry
    // that fails validation or driver loading is deleted before zero returns.
    std::uint32_t load (const std::string& vertexSource, const std::string& fragmentSource)
    {
        if (! enabled())
            return 0;
        const auto path = entryPath (*directory_, keyFor (vertexSource, fragmentSource));
        std::error_code ec;
        if (! std::filesystem::exists (path, ec))
            return 0;
        std::vector<std::uint8_t> bytes;
        std::vector<std::uint8_t> payload;
        std::uint32_t binaryFormat = 0;
        if (! readFile (path, limits_.maximumBytes, bytes)
            || ! decodeEntry (bytes, binaryFormat, payload))
        {
            removeEntry (path);
            return 0;
        }
        const auto program = driver_.createProgram();
        if (program == 0)
            return 0;
        if (! driver_.loadBinary (program, binaryFormat, payload.data(), payload.size()))
        {
            driver_.deleteProgram (program);
            removeEntry (path);
            return 0;
        }
        touchEntry (path);
        return program;
    }

    // Call before linking a program that store() may keep.
    void prepareLink (std::uint32_t program)
    {
        if (enabled())
            driver_.setRetrievableHint (program);
    }

    // Stores the binary of a program that linked successfully. A failure prints
    // one line to stderr and leaves the program as it is.
    void store (const std::string& vertexSource, const std::string& fragmentSource, std::uint32_t program)
    {
        if (! enabled())
            return;
        std::uint32_t binaryFormat = 0;
        std::vector<std::uint8_t> payload;
        std::string error;
        if (! driver_.retrieveBinary (program, binaryFormat, payload))
            error = "the driver returned no program binary";
        else if (payload.empty() || payload.size() > limits_.maximumBytes)
            error = "the program binary is outside the cache size limit";
        else
            error = writeEntry (*directory_, keyFor (vertexSource, fragmentSource), binaryFormat,
                                payload, limits_);
        if (! error.empty())
            std::fprintf (stderr, "gl program cache: entry not written: %s\n", error.c_str());
    }

private:
    std::string keyFor (const std::string& vertexSource, const std::string& fragmentSource) const
    {
        return programKey (vertexSource, fragmentSource, driver_.identity());
    }

    Driver& driver_;
    std::optional<std::filesystem::path> directory_;
    Limits limits_;
};

} // namespace arbitglprogramcache
