# CMake toolchain: cross-compile the Windows video-helper on Linux with the
# Ubuntu mingw-w64 cross compiler, linking either the transitional MSYS2
# dependency closure or the isolated source-built LGPL dependency profile.
# Same target ABI as CI's MSYS2/MINGW64 build (a standalone PE exe).
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(_tc x86_64-w64-mingw32)
# POSIX-threads variants — the helper uses std::thread / std::mutex.
set(CMAKE_C_COMPILER   ${_tc}-gcc-posix)
set(CMAKE_CXX_COMPILER ${_tc}-g++-posix)
set(CMAKE_RC_COMPILER  ${_tc}-windres)

# Search the selected dependency sysroot first, then the mingw toolchain's own sysroot
# (opengl32, the Win32 import libs). ARBIT_MINGW_SYSROOT is set by the build script.
set(CMAKE_FIND_ROOT_PATH "$ENV{ARBIT_MINGW_SYSROOT}/mingw64" "/usr/${_tc}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # use host cmake/ninja/pkg-config
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
