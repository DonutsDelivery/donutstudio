# Finds the OptiX SDK and the CUDA headers it includes. The SDK is headers alone
# and the runtime ships in the NVIDIA driver, so there is no library to find. A
# machine without either configures normally and builds without the OptiX device.

# Where NVIDIA's installers and common manual installs put it. A root may be a glob.
function(arbit_optix_default_roots out_roots)
    set(${out_roots}
        "$ENV{HOME}/NVIDIA-OptiX-SDK-*"
        "/opt/NVIDIA-OptiX-SDK-*"
        "/usr/local/NVIDIA-OptiX-SDK-*"
        "/usr/local/optix"
        "/opt/optix"
        "/opt/nvidia/optix"
        "$ENV{ProgramData}/NVIDIA Corporation/OptiX SDK *"
        PARENT_SCOPE)
endfunction()

function(_arbit_optix_sdk_complete dir out_complete)
    set(complete TRUE)
    foreach(header optix.h optix_stubs.h optix_function_table_definition.h)
        if(NOT EXISTS "${dir}/include/${header}")
            set(complete FALSE)
        endif()
    endforeach()
    set(${out_complete} ${complete} PARENT_SCOPE)
endfunction()

# Sets out_dir to the SDK root, or to empty. out_version is OPTIX_VERSION from
# optix.h, and out_detail says what happened for the status line. An explicit
# directory that holds no SDK finds nothing, because using another SDK than the
# one asked for would hide the mistake. Otherwise the roots are tried in order,
# and the highest version of a glob wins.
function(arbit_find_optix_sdk out_dir out_version out_detail explicit_dir)
    set(found "")
    if(explicit_dir)
        _arbit_optix_sdk_complete("${explicit_dir}" complete)
        if(complete)
            set(found "${explicit_dir}")
        else()
            set(${out_dir} "" PARENT_SCOPE)
            set(${out_version} "" PARENT_SCOPE)
            set(${out_detail} "OptiX_INSTALL_DIR '${explicit_dir}' holds no include/optix.h, optix_stubs.h and optix_function_table_definition.h" PARENT_SCOPE)
            return()
        endif()
    else()
        foreach(root IN LISTS ARGN)
            file(GLOB candidates LIST_DIRECTORIES true "${root}")
            if(candidates)
                list(SORT candidates COMPARE NATURAL ORDER DESCENDING)
            endif()
            foreach(candidate IN LISTS candidates)
                _arbit_optix_sdk_complete("${candidate}" complete)
                if(complete AND NOT found)
                    set(found "${candidate}")
                endif()
            endforeach()
        endforeach()
    endif()

    if(NOT found)
        set(${out_dir} "" PARENT_SCOPE)
        set(${out_version} "" PARENT_SCOPE)
        set(${out_detail} "no SDK in OptiX_INSTALL_DIR or the usual install locations" PARENT_SCOPE)
        return()
    endif()

    file(STRINGS "${found}/include/optix.h" version_line REGEX "^#define OPTIX_VERSION [0-9]+")
    string(REGEX REPLACE "^#define OPTIX_VERSION ([0-9]+).*$" "\\1" version "${version_line}")
    set(${out_dir} "${found}" PARENT_SCOPE)
    set(${out_version} "${version}" PARENT_SCOPE)
    set(${out_detail} "found ${found} (OPTIX_VERSION ${version})" PARENT_SCOPE)
endfunction()

# Where cuda.h can live, best guess first. nvcc on the PATH names the toolkit in
# use, including a distribution's /usr install.
function(arbit_cuda_default_roots out_roots)
    set(roots "$ENV{CUDA_PATH}" "$ENV{CUDA_HOME}")
    find_program(arbit_nvcc nvcc NO_CACHE)
    if(arbit_nvcc)
        get_filename_component(nvcc_real "${arbit_nvcc}" REALPATH)
        get_filename_component(nvcc_bin "${nvcc_real}" DIRECTORY)
        get_filename_component(nvcc_root "${nvcc_bin}" DIRECTORY)
        list(APPEND roots "${nvcc_root}")
    endif()
    list(APPEND roots "/opt/cuda" "/usr/local/cuda")
    set(${out_roots} ${roots} PARENT_SCOPE)
endfunction()

# Sets out_dir to the first <root>/include that holds cuda.h, or to empty. optix.h
# includes cuda.h in host code, so an SDK without it cannot compile.
function(arbit_find_cuda_headers out_dir)
    foreach(root IN LISTS ARGN)
        if(root AND EXISTS "${root}/include/cuda.h")
            set(${out_dir} "${root}/include" PARENT_SCOPE)
            return()
        endif()
    endforeach()
    set(${out_dir} "" PARENT_SCOPE)
endfunction()
