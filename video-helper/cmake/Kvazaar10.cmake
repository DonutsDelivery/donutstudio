# Explicit helper-only candidate. No production encoder selection is changed.
option(ARBIT_BUILD_KVAZAAR10_ADAPTER "Build the isolated helper ten-bit HEVC adapter" OFF)
if(NOT ARBIT_BUILD_KVAZAAR10_ADAPTER)
    return()
endif()
if(CMAKE_VERSION VERSION_LESS 3.25)
    message(FATAL_ERROR "Private Kvazaar 2.3.2 requires CMake 3.25 or newer")
endif()
set(_kvz10_namespace_args --format gnu
    --nm "${CMAKE_NM}" --objcopy "${CMAKE_OBJCOPY}" --ranlib "${CMAKE_RANLIB}")
if(APPLE)
    if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"
       OR NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"
       OR NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        message(FATAL_ERROR "Private Mach-O Kvazaar requires native Apple Silicon and explicit CMAKE_OSX_ARCHITECTURES=arm64")
    endif()
    set(ARBIT_KVAZAAR10_LLVM_VERSION "" CACHE STRING "Recorded LLVM release, at least 20.1.8")
    foreach(_tool NM OBJCOPY RANLIB)
        set(ARBIT_KVAZAAR10_LLVM_${_tool} "" CACHE FILEPATH "Explicit recorded LLVM ${_tool} executable")
    endforeach()
    set(_kvz10_namespace_args --format macho-arm64
        --nm "${ARBIT_KVAZAAR10_LLVM_NM}" --objcopy "${ARBIT_KVAZAAR10_LLVM_OBJCOPY}"
        --ranlib "${ARBIT_KVAZAAR10_LLVM_RANLIB}" --llvm-version "${ARBIT_KVAZAAR10_LLVM_VERSION}")
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace_kvazaar10.py"
            ${_kvz10_namespace_args} --check-tools
        RESULT_VARIABLE _kvz10_tool_result OUTPUT_VARIABLE _kvz10_tools ERROR_VARIABLE _kvz10_tool_error)
    if(NOT _kvz10_tool_result EQUAL 0)
        message(FATAL_ERROR "Private Mach-O Kvazaar tool admission failed: ${_kvz10_tool_error}")
    endif()
    message(STATUS "Private Mach-O Kvazaar recorded LLVM tools: ${_kvz10_tools}")
elseif(MSVC OR NOT CMAKE_NM OR NOT CMAKE_OBJCOPY OR NOT CMAKE_RANLIB)
    message(FATAL_ERROR "Private Kvazaar needs verified GNU-compatible nm/objcopy/ranlib; this toolchain is not admitted")
endif()
include(FetchContent)
FetchContent_Declare(arbit_kvazaar10_source
    URL https://codeload.github.com/ultravideo/kvazaar/tar.gz/refs/tags/v2.3.2
    URL_HASH SHA256=ddd0038696631ca5368d8e40efee36d2bbb805854b9b1dda8b12ea9b397ea951)
FetchContent_GetProperties(arbit_kvazaar10_source)
if(NOT arbit_kvazaar10_source_POPULATED)
    FetchContent_Populate(arbit_kvazaar10_source)
endif()
file(SHA256 "${arbit_kvazaar10_source_SOURCE_DIR}/src/kvazaar.h" ARBIT_KVAZAAR10_HEADER_SHA)
if(NOT ARBIT_KVAZAAR10_HEADER_SHA STREQUAL "fc875992577bdd54eb20cb28057e673293f769246cc8a1d6f0b1d176178df361")
    message(FATAL_ERROR "Private Kvazaar header differs from the verified 2.3.2 pin")
endif()
function(arbit_add_kvazaar10_source)
    # Function scope keeps upstream's general option names out of helper defaults.
    set(BUILD_SHARED_LIBS OFF)
    set(BUILD_TESTS OFF)
    set(BUILD_KVAZAAR_BINARY OFF)
    set(USE_CRYPTO OFF)
    add_subdirectory("${arbit_kvazaar10_source_SOURCE_DIR}"
                     "${arbit_kvazaar10_source_BINARY_DIR}" EXCLUDE_FROM_ALL)
endfunction()
arbit_add_kvazaar10_source()
target_sources(kvazaar PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/kvazaar10_depth.c")
target_compile_definitions(kvazaar PRIVATE KVZ_BIT_DEPTH=10)
set_target_properties(kvazaar PROPERTIES POSITION_INDEPENDENT_CODE ON)
set(ARBIT_KVAZAAR10_ARCHIVE "${CMAKE_CURRENT_BINARY_DIR}/kvazaar10/libdonut-kvazaar10.a")
add_custom_command(OUTPUT "${ARBIT_KVAZAAR10_ARCHIVE}"
    BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/kvazaar10/libdonut-kvazaar10.symbols"
               "${CMAKE_CURRENT_BINARY_DIR}/kvazaar10/libdonut-kvazaar10.namespace.json"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace_kvazaar10.py"
        ${_kvz10_namespace_args}
        --input "$<TARGET_FILE:kvazaar>" --output "${ARBIT_KVAZAAR10_ARCHIVE}"
    DEPENDS kvazaar "${CMAKE_CURRENT_SOURCE_DIR}/cmake/namespace_kvazaar10.py"
    VERBATIM)
add_custom_target(arbit-kvazaar10-namespace DEPENDS "${ARBIT_KVAZAAR10_ARCHIVE}")
add_library(arbit-kvazaar10-private STATIC IMPORTED)
set_target_properties(arbit-kvazaar10-private PROPERTIES IMPORTED_LOCATION "${ARBIT_KVAZAAR10_ARCHIVE}")
add_dependencies(arbit-kvazaar10-private arbit-kvazaar10-namespace)
find_package(Threads REQUIRED)
add_library(arbit-kvazaar10-adapter STATIC EXCLUDE_FROM_ALL src/kvazaar10_encoder.cpp)
target_include_directories(arbit-kvazaar10-adapter PRIVATE "${arbit_kvazaar10_source_SOURCE_DIR}/src")
target_compile_definitions(arbit-kvazaar10-adapter PRIVATE KVZ_BIT_DEPTH=10 KVZ_STATIC_LIB=1)
target_link_libraries(arbit-kvazaar10-adapter PRIVATE arbit-kvazaar10-private PkgConfig::AV Threads::Threads)
if(UNIX)
    target_link_libraries(arbit-kvazaar10-adapter PRIVATE m)
endif()

# Fake API leaf fixture exercises ownership and delayed packets without an encoder.
add_executable(arbit-kvazaar10-adapter-tests EXCLUDE_FROM_ALL
    tests/kvazaar10_encoder_tests.cpp src/kvazaar10_encoder.cpp)
target_include_directories(arbit-kvazaar10-adapter-tests PRIVATE src "${arbit_kvazaar10_source_SOURCE_DIR}/src")
target_compile_definitions(arbit-kvazaar10-adapter-tests PRIVATE KVZ_BIT_DEPTH=10 KVZ_STATIC_LIB=1)
target_link_libraries(arbit-kvazaar10-adapter-tests PRIVATE PkgConfig::AV)
add_test(NAME kvazaar10_adapter COMMAND arbit-kvazaar10-adapter-tests)
add_test(NAME kvazaar10_namespace
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_kvazaar10_namespace.py")

if(APPLE)
    option(ARBIT_KVAZAAR10_BUILD_LINK_SMOKE "Prepare real Main10 and FFmpeg eight-bit dylib coexistence smoke" OFF)
    if(ARBIT_KVAZAAR10_BUILD_LINK_SMOKE)
        get_target_property(_kvz10_av_link_options PkgConfig::AV INTERFACE_LINK_OPTIONS)
        foreach(_flags CMAKE_EXE_LINKER_FLAGS CMAKE_EXE_LINKER_FLAGS_DEBUG
                CMAKE_EXE_LINKER_FLAGS_RELEASE CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO
                CMAKE_EXE_LINKER_FLAGS_MINSIZEREL _kvz10_av_link_options)
            if("${${_flags}}" MATCHES "flat_namespace")
                message(FATAL_ERROR "Main10 link smoke rejects flat-namespace linker flags")
            endif()
        endforeach()
        set(ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB "" CACHE FILEPATH "FFmpeg profile's ordinary eight-bit Kvazaar dylib")
        if(NOT IS_ABSOLUTE "${ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB}"
           OR NOT EXISTS "${ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB}"
           OR NOT ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB MATCHES "\\.dylib$")
            message(FATAL_ERROR "Main10 link smoke requires the explicit absolute FFmpeg eight-bit Kvazaar dylib")
        endif()
        find_library(_kvz10_smoke_avcodec NAMES avcodec PATHS ${AV_LIBRARY_DIRS} NO_DEFAULT_PATH REQUIRED)
        if(NOT _kvz10_smoke_avcodec MATCHES "\\.dylib$")
            message(FATAL_ERROR "Main10 link smoke requires FFmpeg's shared libavcodec dylib")
        endif()
        add_executable(arbit-kvazaar10-link-smoke EXCLUDE_FROM_ALL tests/kvazaar10_link_smoke.cpp)
        target_include_directories(arbit-kvazaar10-link-smoke PRIVATE src)
        target_link_libraries(arbit-kvazaar10-link-smoke PRIVATE arbit-kvazaar10-adapter
            PkgConfig::AV "${ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB}")
        target_link_options(arbit-kvazaar10-link-smoke PRIVATE "LINKER:-twolevel_namespace")
        add_test(NAME kvazaar10_macho_link_smoke
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/kvazaar10_macho_link_smoke.py"
                --executable "$<TARGET_FILE:arbit-kvazaar10-link-smoke>"
                --archive "${ARBIT_KVAZAAR10_ARCHIVE}" --avcodec "${_kvz10_smoke_avcodec}"
                --kvazaar "${ARBIT_KVAZAAR10_FFMPEG_KVAZAAR_DYLIB}"
                --nm "${ARBIT_KVAZAAR10_LLVM_NM}")
        set_tests_properties(kvazaar10_macho_link_smoke PROPERTIES TIMEOUT 120 RUN_SERIAL TRUE)
    endif()
endif()
