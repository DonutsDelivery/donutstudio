// Contract tests for the fixture program binary cache. None of these tests
// touch GL. A scripted driver stands in for glProgramBinary and the other
// entry points, so the file logic is checked on every platform.

#include "../src/gpu_backend/gl_program_binary_cache.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace fs = std::filesystem;
namespace cache = arbitglprogramcache;

namespace
{
int failures = 0;

void check (bool value, const char* message)
{
    if (! value)
    {
        ++failures;
        std::fprintf (stderr, "FAIL: %s\n", message);
    }
}

// Records every call the cache makes. Tests use it to assert that a disabled
// or invalid cache never reaches the driver.
class FakeDriver final : public cache::Driver
{
public:
    int formats = 1;
    bool acceptBinaries = true;
    cache::DriverIdentity identityValue { "vendor", "renderer", "version" };
    std::uint32_t retrieveFormat = 0x8e01;
    std::vector<std::uint8_t> retrieveBytes { 1, 2, 3, 4, 5, 6, 7, 8 };
    std::uint32_t nextProgram = 1;
    int hints = 0;
    int loads = 0;
    int deletes = 0;
    int retrieves = 0;
    std::uint32_t loadedFormat = 0;
    std::vector<std::uint8_t> loadedBytes;

    int binaryFormatCount() const override { return formats; }
    cache::DriverIdentity identity() const override { return identityValue; }
    std::uint32_t createProgram() override { return nextProgram++; }
    void deleteProgram (std::uint32_t) override { ++deletes; }
    void setRetrievableHint (std::uint32_t) override { ++hints; }

    bool loadBinary (std::uint32_t, std::uint32_t binaryFormat,
                     const std::uint8_t* data, std::size_t size) override
    {
        ++loads;
        loadedFormat = binaryFormat;
        loadedBytes.assign (data, data + size);
        return acceptBinaries;
    }

    bool retrieveBinary (std::uint32_t, std::uint32_t& binaryFormat,
                         std::vector<std::uint8_t>& payload) override
    {
        ++retrieves;
        if (retrieveBytes.empty())
            return false;
        binaryFormat = retrieveFormat;
        payload = retrieveBytes;
        return true;
    }
};

void removeTree (const fs::path& target)
{
    std::error_code ec;
    (void) fs::remove_all (target, ec);
}

// A path under the system temporary directory that does not exist yet.
fs::path freshPath (const std::string& name)
{
    static int sequence = 0;
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path path = fs::temp_directory_path() / ("arbit-gl-program-cache-" + name + "-"
        + std::to_string (stamp) + "-" + std::to_string (++sequence));
    removeTree (path);
    return path;
}

bool isHexDigest (const std::string& text)
{
    if (text.size() != 64)
        return false;
    for (const char c : text)
    {
        if (! ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f')))
            return false;
    }
    return true;
}

// Moves an entry's modification time to the given age before now.
void ageEntry (const fs::path& file, std::chrono::hours age)
{
    std::error_code ec;
    fs::last_write_time (file, fs::file_time_type::clock::now() - age, ec);
}

void testProgramKey()
{
    const cache::DriverIdentity identity { "vendor", "renderer", "version" };
    const auto key = cache::programKey ("vertex", "fragment", identity);
    check (isHexDigest (key), "a program key is 64 lowercase hex digits");
    check (key == cache::programKey ("vertex", "fragment", identity), "a program key is stable for equal inputs");
    check (key != cache::programKey ("vertex ", "fragment", identity), "a vertex source change changes the key");
    check (key != cache::programKey ("vertex", "fragment ", identity), "a fragment source change changes the key");
    check (cache::programKey ("ab", "c", identity) != cache::programKey ("a", "bc", identity),
           "length prefixes keep adjacent sources apart");
    check (key != cache::programKey ("vertex", "fragment", { "other", "renderer", "version" }),
           "a vendor change changes the key");
    check (key != cache::programKey ("vertex", "fragment", { "vendor", "other", "version" }),
           "a renderer change changes the key");
    check (key != cache::programKey ("vertex", "fragment", { "vendor", "renderer", "other" }),
           "a version change changes the key");
    check (key != cache::programKey ("vertex", "fragment", identity, cache::kFormatVersion + 1),
           "a format version change changes the key");
}

void testEntryCodec()
{
    const std::vector<std::uint8_t> payload { 9, 8, 7, 6, 5 };
    const auto bytes = cache::encodeEntry (0x1234, payload);
    std::uint32_t format = 0;
    std::vector<std::uint8_t> decoded;
    check (bytes.size() == cache::kHeaderBytes + payload.size(), "an entry is the header followed by the payload");
    check (cache::decodeEntry (bytes, format, decoded), "an encoded entry decodes");
    check (format == 0x1234 && decoded == payload, "a decoded entry returns its format and payload");

    auto headerOnly = bytes;
    headerOnly.resize (cache::kHeaderBytes);
    check (! cache::decodeEntry (headerOnly, format, decoded), "an entry without a payload is rejected");

    auto magic = bytes;
    magic[0] = 'X';
    check (! cache::decodeEntry (magic, format, decoded), "a wrong magic is rejected");

    auto version = bytes;
    version[8] ^= 0x01;
    check (! cache::decodeEntry (version, format, decoded), "a wrong format version is rejected");

    auto extra = bytes;
    extra.push_back (0);
    check (! cache::decodeEntry (extra, format, decoded), "a payload length mismatch is rejected");

    auto corruptPayload = bytes;
    corruptPayload.back() ^= 0x01;
    check (! cache::decodeEntry (corruptPayload, format, decoded), "a payload digest mismatch is rejected");

    auto corruptDigest = bytes;
    corruptDigest[24] = static_cast<std::uint8_t> (corruptDigest[24] == '0' ? '1' : '0');
    check (! cache::decodeEntry (corruptDigest, format, decoded), "an altered stored digest is rejected");
}

void testStoreThenLoad()
{
    const fs::path directory = freshPath ("round-trip");
    FakeDriver writer;
    cache::ProgramBinaryCache storing (writer, directory);
    check (storing.enabled(), "a directory and binary formats enable the cache");
    storing.prepareLink (7);
    check (writer.hints == 1, "prepareLink sets the retrievable hint once");
    storing.store ("vertex", "fragment", 7);
    check (writer.retrieves == 1, "store retrieves the binary once");

    const auto file = cache::entryPath (directory, cache::programKey ("vertex", "fragment", writer.identity()));
    check (fs::exists (file), "store publishes the entry under its key");

    FakeDriver reader;
    cache::ProgramBinaryCache loading (reader, directory);
    const auto program = loading.load ("vertex", "fragment");
    check (program != 0, "an identical pair loads a program");
    check (reader.loads == 1 && reader.loadedFormat == writer.retrieveFormat && reader.loadedBytes == writer.retrieveBytes,
           "the driver receives the stored format and payload");
    check (reader.deletes == 0 && fs::exists (file), "a hit keeps the entry");

    FakeDriver other;
    cache::ProgramBinaryCache different (other, directory);
    check (different.load ("vertex", "fragment changed") == 0 && other.loads == 0,
           "a different fragment source misses without reaching the driver");
    removeTree (directory);
}

void testCorruptEntryIsRemoved()
{
    const fs::path directory = freshPath ("corrupt");
    (void) fs::create_directories (directory);
    FakeDriver driver;
    const auto file = cache::entryPath (directory, cache::programKey ("vertex", "fragment", driver.identity()));
    {
        std::ofstream out (file, std::ios::binary);
        out << "not a cache entry";
    }
    cache::ProgramBinaryCache binary (driver, directory);
    check (binary.load ("vertex", "fragment") == 0, "a corrupt entry misses");
    check (driver.loads == 0, "a corrupt entry never reaches the driver");
    check (! fs::exists (file), "a corrupt entry is removed");
    removeTree (directory);
}

void testDriverRejectionRemovesEntry()
{
    const fs::path directory = freshPath ("rejected");
    FakeDriver writer;
    cache::ProgramBinaryCache storing (writer, directory);
    storing.store ("vertex", "fragment", 3);

    FakeDriver reader;
    reader.acceptBinaries = false;
    cache::ProgramBinaryCache loading (reader, directory);
    check (loading.load ("vertex", "fragment") == 0, "a driver that rejects the binary misses");
    check (reader.loads == 1 && reader.deletes == 1, "the rejected program is deleted once");
    check (! fs::exists (cache::entryPath (directory, cache::programKey ("vertex", "fragment", reader.identity()))),
           "a rejected entry is removed");
    removeTree (directory);
}

void testDisabledCache()
{
    const fs::path directory = freshPath ("disabled");

    FakeDriver noFormats;
    noFormats.formats = 0;
    cache::ProgramBinaryCache unsupported (noFormats, directory);
    check (! unsupported.enabled(), "a driver without binary formats disables the cache");
    unsupported.prepareLink (1);
    unsupported.store ("vertex", "fragment", 1);
    check (unsupported.load ("vertex", "fragment") == 0, "a disabled cache loads nothing");
    check (noFormats.hints == 0 && noFormats.retrieves == 0 && noFormats.loads == 0,
           "a disabled cache never reaches the driver");
    check (! fs::exists (directory), "a disabled cache never creates its directory");

    FakeDriver noDirectory;
    cache::ProgramBinaryCache unlocated (noDirectory, std::nullopt);
    check (! unlocated.enabled(), "a missing directory disables the cache");
    unlocated.prepareLink (1);
    unlocated.store ("vertex", "fragment", 1);
    check (unlocated.load ("vertex", "fragment") == 0, "a cache without a directory loads nothing");
    check (noDirectory.hints == 0 && noDirectory.retrieves == 0 && noDirectory.loads == 0,
           "a cache without a directory never reaches the driver");
}

void testStateLine()
{
    const cache::EntryPointReport all { true, true, true };
    const cache::EntryPointReport missingGet { true, false, true };
    check (cache::stateLine (fs::path ("/cache"), 1, all)
               == "gl program cache: enabled at /cache (binaryFormats=1)",
           "an enabled cache reports its directory and format count");
    check (cache::stateLine (std::nullopt, 0, missingGet)
               == "gl program cache: disabled (directory=none, binaryFormats=0, "
                  "entryPoints=ProgramBinary y,GetProgramBinary n,ProgramParameteri y)",
           "a cache without a directory reports none and the entry point flags");
    check (cache::stateLine (fs::path ("/cache"), 0, cache::EntryPointReport {})
               == "gl program cache: disabled (directory=/cache, binaryFormats=0, "
                  "entryPoints=ProgramBinary n,GetProgramBinary n,ProgramParameteri n)",
           "a driver without binary formats reports the directory and zero formats");

    FakeDriver driver;
    cache::ProgramBinaryCache enabledCache (driver, fs::path ("/cache"));
    check (enabledCache.directory() == fs::path ("/cache"), "the cache exposes its directory");
    cache::ProgramBinaryCache unlocated (driver, std::nullopt);
    check (! unlocated.directory().has_value(), "the cache exposes a missing directory");
}

void testStoreRejectsUnusableBinaries()
{
    const fs::path directory = freshPath ("unusable");
    FakeDriver driver;
    driver.retrieveBytes.clear();
    cache::ProgramBinaryCache empty (driver, directory);
    empty.store ("vertex", "fragment", 1);
    check (! fs::exists (directory), "an empty binary is not written");

    driver.retrieveBytes = { 1, 2, 3, 4, 5, 6, 7, 8 };
    cache::Limits tight;
    tight.maximumBytes = 4;
    cache::ProgramBinaryCache oversized (driver, directory, tight);
    oversized.store ("vertex", "fragment", 1);
    check (! fs::exists (cache::entryPath (directory, cache::programKey ("vertex", "fragment", driver.identity()))),
           "a binary over the size limit is not written");
    removeTree (directory);
}

void testAtomicWriteAndFailureReport()
{
    const fs::path directory = freshPath ("temporary");
    (void) fs::create_directories (directory);
    const fs::path stale = directory / "abandoned.1.1.tmp";
    const fs::path recent = directory / "in-progress.1.2.tmp";
    {
        std::ofstream out (stale);
        out << "partial";
    }
    {
        std::ofstream out (recent);
        out << "partial";
    }
    ageEntry (stale, std::chrono::hours (2));

    const auto error = cache::writeEntry (directory, "abc", 5, std::vector<std::uint8_t> { 1, 2, 3 }, cache::Limits {});
    check (error.empty(), "writeEntry succeeds in a writable directory");
    check (fs::exists (cache::entryPath (directory, "abc")), "writeEntry publishes the entry");
    check (! fs::exists (stale), "a temporary file older than an hour is removed");
    check (fs::exists (recent), "a recent temporary file is kept");
    std::size_t temporaries = 0;
    for (const auto& item : fs::directory_iterator (directory))
    {
        if (item.path().extension().string() == ".tmp")
            ++temporaries;
    }
    check (temporaries == 1, "a successful write leaves no temporary file of its own");
    removeTree (directory);

    const fs::path blocker = freshPath ("blocker");
    {
        std::ofstream out (blocker);
        out << "file";
    }
    check (! cache::writeEntry (blocker / "child", "abc", 1, std::vector<std::uint8_t> (4, 1), cache::Limits {}).empty(),
           "a cache directory under a regular file reports the failure");
    removeTree (blocker);
}

void testEvictionByCount()
{
    const fs::path directory = freshPath ("count");
    const std::vector<std::uint8_t> payload (16, 0x5a);
    cache::Limits roomy;
    check (cache::writeEntry (directory, "aaa", 1, payload, roomy).empty(), "the first entry is written");
    check (cache::writeEntry (directory, "bbb", 1, payload, roomy).empty(), "the second entry is written");
    check (cache::writeEntry (directory, "ccc", 1, payload, roomy).empty(), "the third entry is written");
    ageEntry (cache::entryPath (directory, "aaa"), std::chrono::hours (3));
    ageEntry (cache::entryPath (directory, "bbb"), std::chrono::hours (2));
    ageEntry (cache::entryPath (directory, "ccc"), std::chrono::hours (1));

    cache::Limits two;
    two.maximumEntries = 2;
    cache::evictEntries (directory, cache::entryPath (directory, "aaa"), two);
    check (fs::exists (cache::entryPath (directory, "aaa")),
           "the entry passed as keep survives even when it is the oldest");
    check (! fs::exists (cache::entryPath (directory, "bbb")),
           "the next oldest entry is removed to meet the count limit");
    check (fs::exists (cache::entryPath (directory, "ccc")), "the newest entry survives count eviction");

    check (cache::writeEntry (directory, "ddd", 1, payload, two).empty(), "a write under the count limit succeeds");
    check (! fs::exists (cache::entryPath (directory, "aaa")), "a write evicts the oldest entry");
    check (fs::exists (cache::entryPath (directory, "ccc")) && fs::exists (cache::entryPath (directory, "ddd")),
           "the two newest entries remain after a write");
    removeTree (directory);
}

void testEvictionByBytes()
{
    const fs::path directory = freshPath ("bytes");
    const std::vector<std::uint8_t> payload (100, 0x33);
    cache::Limits oneEntry;
    oneEntry.maximumBytes = cache::kHeaderBytes + payload.size() + 12;
    check (cache::writeEntry (directory, "aaa", 1, payload, oneEntry).empty(), "the first entry fits the byte limit");
    ageEntry (cache::entryPath (directory, "aaa"), std::chrono::hours (3));
    check (cache::writeEntry (directory, "bbb", 1, payload, oneEntry).empty(), "the second entry is written");
    check (! fs::exists (cache::entryPath (directory, "aaa")), "the oldest entry is evicted to meet the byte limit");
    check (fs::exists (cache::entryPath (directory, "bbb")), "the newest entry survives byte eviction");
    removeTree (directory);
}

void testResolveDirectory()
{
    check (! cache::resolveDirectory ("", "/xdg", "/home/user", "C:\\Local").has_value(),
           "an empty override disables the cache");
    check (cache::resolveDirectory ("/custom/cache", "/xdg", "/home/user", nullptr) == fs::path ("/custom/cache"),
           "an override is used as the directory");
    check (! cache::resolveDirectory (nullptr, nullptr, nullptr, nullptr).has_value(),
           "no base directory disables the cache");
#if defined(_WIN32)
    check (cache::resolveDirectory (nullptr, "/xdg", "/home/user", "C:\\Local")
               == fs::path ("C:\\Local") / "DonutStudio" / "gl-program-cache",
           "Windows uses LOCALAPPDATA");
    check (! cache::resolveDirectory (nullptr, "/xdg", "/home/user", nullptr).has_value(),
           "Windows ignores XDG_CACHE_HOME and HOME");
#elif defined(__APPLE__)
    check (cache::resolveDirectory (nullptr, "/xdg", "/home/user", nullptr)
               == fs::path ("/home/user") / "Library" / "Caches" / "DonutStudio" / "gl-program-cache",
           "macOS uses HOME/Library/Caches");
#else
    check (cache::resolveDirectory (nullptr, "/xdg", "/home/user", nullptr)
               == fs::path ("/xdg") / "DonutStudio" / "gl-program-cache",
           "XDG_CACHE_HOME is preferred on Linux");
    check (cache::resolveDirectory (nullptr, nullptr, "/home/user", nullptr)
               == fs::path ("/home/user") / ".cache" / "DonutStudio" / "gl-program-cache",
           "HOME/.cache is the Linux fallback");
    check (! cache::resolveDirectory (nullptr, "", "", nullptr).has_value(),
           "empty base variables disable the cache on Linux");
#endif
}

#if !defined(_WIN32)
void testEnvironmentOverride()
{
    const fs::path directory = freshPath ("environment");
    (void) setenv (cache::kEnvironmentVariable, directory.c_str(), 1);
    check (cache::cacheDirectory() == directory, "ARBIT_GL_PROGRAM_CACHE_DIR selects the directory");
    (void) setenv (cache::kEnvironmentVariable, "", 1);
    check (! cache::cacheDirectory().has_value(), "an empty ARBIT_GL_PROGRAM_CACHE_DIR disables the cache");
    (void) unsetenv (cache::kEnvironmentVariable);
}
#endif

} // namespace

int main()
{
    testProgramKey();
    testEntryCodec();
    testStoreThenLoad();
    testCorruptEntryIsRemoved();
    testDriverRejectionRemovesEntry();
    testDisabledCache();
    testStateLine();
    testStoreRejectsUnusableBinaries();
    testAtomicWriteAndFailureReport();
    testEvictionByCount();
    testEvictionByBytes();
    testResolveDirectory();
#if !defined(_WIN32)
    testEnvironmentOverride();
#endif
    if (failures != 0)
    {
        std::fprintf (stderr, "%d gl program binary cache checks failed\n", failures);
        return 1;
    }
    std::fprintf (stderr, "gl program binary cache checks passed\n");
    return 0;
}
