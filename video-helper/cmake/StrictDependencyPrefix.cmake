# An explicit source prefix admits only its audited FFmpeg closure.
set(ARBIT_VIDEO_DEPENDENCY_PREFIX "" CACHE PATH "Audited source-built video codec prefix")

function(arbit_find_prefix_av)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/strict_dependency_prefix.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}/.."
            --prefix "${ARBIT_VIDEO_DEPENDENCY_PREFIX}"
            --system "${CMAKE_SYSTEM_NAME}" --processor "${CMAKE_SYSTEM_PROCESSOR}"
            "--osx-architectures=${CMAKE_OSX_ARCHITECTURES}" "--mingw=${MINGW}"
        RESULT_VARIABLE _receipt_result ERROR_VARIABLE _receipt_error)
    if(NOT _receipt_result EQUAL 0)
        message(FATAL_ERROR "Video dependency prefix rejected: ${_receipt_error}")
    endif()

    # Scope pkg-config to AV only. Lua/GLFW/NCNN keep their normal discovery.
    foreach(_name PKG_CONFIG_PATH PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR)
        if(DEFINED ENV{${_name}})
            set(_saved_${_name} "$ENV{${_name}}")
            set(_had_${_name} TRUE)
        endif()
    endforeach()
    set(ENV{PKG_CONFIG_PATH} "")
    set(ENV{PKG_CONFIG_LIBDIR} "${ARBIT_VIDEO_DEPENDENCY_PREFIX}/lib/pkgconfig")
    set(ENV{PKG_CONFIG_SYSROOT_DIR} "")
    # Clear a previous configure's discovery, including find_library cache entries.
    get_cmake_property(_cache_variables CACHE_VARIABLES)
    foreach(_variable IN LISTS _cache_variables)
        if(_variable MATCHES "^(ARBIT_PREFIX_AV_|__pkg_config_checked_ARBIT_PREFIX_AV|pkgcfg_lib_ARBIT_PREFIX_AV_)")
            unset(${_variable} CACHE)
        endif()
    endforeach()
    pkg_check_modules(ARBIT_PREFIX_AV QUIET
        NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH
        libavformat libavcodec libavutil libavdevice libswscale libswresample libavfilter)
    set(_binding_args)
    foreach(_module libavformat libavcodec libavutil libavdevice libswscale libswresample libavfilter)
        unset(_pc_dir)
        pkg_get_variable(_pc_dir ${_module} pcfiledir)
        list(APPEND _binding_args --pkg-dir "${_pc_dir}")
    endforeach()
    foreach(_name PKG_CONFIG_PATH PKG_CONFIG_LIBDIR PKG_CONFIG_SYSROOT_DIR)
        if(_had_${_name})
            set(ENV{${_name}} "${_saved_${_name}}")
        else()
            unset(ENV{${_name}})
        endif()
    endforeach()
    if(NOT ARBIT_PREFIX_AV_FOUND)
        message(FATAL_ERROR "Explicit video dependency prefix has no complete libav pkg-config closure; system fallback is forbidden")
    endif()
    foreach(_include IN LISTS ARBIT_PREFIX_AV_INCLUDE_DIRS)
        list(APPEND _binding_args --include-dir "${_include}")
    endforeach()
    foreach(_dir IN LISTS ARBIT_PREFIX_AV_LIBRARY_DIRS)
        list(APPEND _binding_args --library-dir "${_dir}")
    endforeach()
    set(_links)
    foreach(_name IN LISTS ARBIT_PREFIX_AV_LIBRARIES)
        if(NOT _name MATCHES "^(avformat|avcodec|avutil|avdevice|avfilter|swscale|swresample)$")
            message(FATAL_ERROR "Explicit libav prefix has an unreviewed link dependency: ${_name}")
        endif()
        # Cross-compilation roots must not redirect this explicit prefix.
        unset(_prefix_library CACHE)
        unset(_prefix_library)
        find_library(_prefix_library NAMES "${_name}"
            PATHS "${ARBIT_VIDEO_DEPENDENCY_PREFIX}/lib"
            NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH)
        if(NOT _prefix_library)
            message(FATAL_ERROR "Explicit libav prefix is missing shared library ${_name}")
        endif()
        list(APPEND _links "${_prefix_library}")
        list(APPEND _binding_args --library "${_prefix_library}")
    endforeach()
    unset(_prefix_library CACHE)
    # Additional -L/-l, response files or linker scripts may escape admission.
    if(ARBIT_PREFIX_AV_CFLAGS_OTHER OR ARBIT_PREFIX_AV_LDFLAGS_OTHER)
        message(FATAL_ERROR "Explicit libav prefix has unreviewed compiler/linker options")
    endif()
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/strict_dependency_prefix.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}/.."
            --prefix "${ARBIT_VIDEO_DEPENDENCY_PREFIX}"
            --system "${CMAKE_SYSTEM_NAME}" --processor "${CMAKE_SYSTEM_PROCESSOR}"
            "--osx-architectures=${CMAKE_OSX_ARCHITECTURES}" "--mingw=${MINGW}"
            --binding ${_binding_args}
        RESULT_VARIABLE _binding_result ERROR_VARIABLE _binding_error)
    if(NOT _binding_result EQUAL 0)
        message(FATAL_ERROR "Video dependency binding rejected: ${_binding_error}")
    endif()
    # A regular interface keeps the audited root as -I. Imported targets make
    # it -isystem, where an earlier generic Homebrew root can replace libav.
    add_library(arbit-prefix-av INTERFACE)
    add_library(PkgConfig::AV ALIAS arbit-prefix-av)
    set_target_properties(arbit-prefix-av PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARBIT_PREFIX_AV_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${_links}")
    if(NOT CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
        message(FATAL_ERROR "Explicit libav prefix requires a compile-command database for header admission")
    endif()
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)
    cmake_language(DEFER CALL arbit_check_prefix_av_consumers)
endfunction()

# Run before any current AV consumer compiles, using its generated compiler
# command rather than treating declared include directories as consumed inputs.
function(arbit_check_prefix_av_consumers)
    get_property(_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
    set(_consumers)
    foreach(_target IN LISTS _targets)
        get_target_property(_links "${_target}" LINK_LIBRARIES)
        if("PkgConfig::AV" IN_LIST _links)
            list(APPEND _consumers "${_target}")
        endif()
    endforeach()
    if(NOT _consumers)
        message(FATAL_ERROR "Explicit libav prefix has no admitted consumers")
    endif()
    string(REPLACE ";" "\n" _consumer_lines "${_consumers}")
    file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/arbit-prefix-av-consumers.txt"
        CONTENT "${_consumer_lines}\n")
    add_custom_target(arbit-prefix-av-header-check
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/strict_dependency_prefix.py"
            --root "${CMAKE_CURRENT_SOURCE_DIR}/.."
            --prefix "${ARBIT_VIDEO_DEPENDENCY_PREFIX}"
            --system "${CMAKE_SYSTEM_NAME}" --processor "${CMAKE_SYSTEM_PROCESSOR}"
            "--osx-architectures=${CMAKE_OSX_ARCHITECTURES}" "--mingw=${MINGW}"
            --compile-build "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Admitting actual compiler-selected libav headers"
        VERBATIM)
    foreach(_target IN LISTS _consumers)
        add_dependencies("${_target}" arbit-prefix-av-header-check)
    endforeach()
endfunction()
