#pragma once

#include <cstddef>
#include <string>

/**
 * Names of POSIX shared memory and semaphores, free of JUCE so the video helper
 * and the plugin host worker can include it.
 *
 * Everywhere except the sandboxed Mac App Store build, POSIX wants a leading
 * slash. The Mac sandbox wants the opposite. A name must start with the app
 * group id and a slash, so the sandbox rejects "/A5LV92TMQA.ds/x".
 * posixObjectName() adds the slash only when the name does not already start
 * with the group prefix.
 *
 * macOS caps these names at 31 characters. Callers on the Mac build use the
 * budget below to keep prefix and base name inside it.
 */
namespace ipcname
{

inline constexpr const char* macAppGroupPrefix = "A5LV92TMQA.ds/";
inline constexpr std::size_t macAppGroupPrefixLength = 14;
inline constexpr std::size_t macNameLimit = 31;

/** Characters left for the base name after the app group prefix. */
inline constexpr std::size_t macBaseNameBudget = macNameLimit - macAppGroupPrefixLength;

inline bool hasMacAppGroupPrefix (const std::string& name)
{
    return name.compare (0, macAppGroupPrefixLength, macAppGroupPrefix) == 0;
}

/** The string to pass to shm_open, sem_open, shm_unlink and sem_unlink. */
inline std::string posixObjectName (const std::string& name)
{
    return hasMacAppGroupPrefix (name) ? name : "/" + name;
}

} // namespace ipcname
