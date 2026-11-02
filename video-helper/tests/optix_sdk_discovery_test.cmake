# Run with cmake -DWORK_DIR=<dir> -P. Discovery of the OptiX SDK against fake
# directory layouts. No real SDK is involved.

cmake_minimum_required(VERSION 3.22)

include("${CMAKE_CURRENT_LIST_DIR}/../cmake/OptiXDiscovery.cmake")

if(NOT WORK_DIR)
    message(FATAL_ERROR "pass -DWORK_DIR=<dir>")
endif()
file(REMOVE_RECURSE "${WORK_DIR}")

function(fake_sdk dir version)
    file(WRITE "${dir}/include/optix.h" "#pragma once\n#define OPTIX_VERSION ${version}\n")
    file(WRITE "${dir}/include/optix_stubs.h" "")
    file(WRITE "${dir}/include/optix_function_table_definition.h" "")
endfunction()

function(expect name actual wanted)
    if(NOT "${actual}" STREQUAL "${wanted}")
        message(FATAL_ERROR "${name}: got '${actual}', wanted '${wanted}'")
    endif()
endfunction()

set(empty "${WORK_DIR}/empty")
file(MAKE_DIRECTORY "${empty}")

# Nothing anywhere: not found, and not an error.
arbit_find_optix_sdk(dir version detail "" "${empty}/optix" "${empty}/NVIDIA-OptiX-SDK-*")
expect("no SDK" "${dir}" "")
expect("no SDK has no version" "${version}" "")
if(NOT detail MATCHES "no SDK")
    message(FATAL_ERROR "no SDK detail: ${detail}")
endif()

# An explicit directory wins over a root that also holds an SDK.
fake_sdk("${WORK_DIR}/explicit" 90000)
fake_sdk("${WORK_DIR}/root" 70000)
arbit_find_optix_sdk(dir version detail "${WORK_DIR}/explicit" "${WORK_DIR}/root")
expect("explicit directory" "${dir}" "${WORK_DIR}/explicit")
expect("version comes from optix.h" "${version}" "90000")

# An explicit directory without an SDK names it and does not fall through.
arbit_find_optix_sdk(dir version detail "${empty}" "${WORK_DIR}/root")
expect("explicit directory without an SDK" "${dir}" "")
if(NOT detail MATCHES "OptiX_INSTALL_DIR '")
    message(FATAL_ERROR "explicit miss detail: ${detail}")
endif()

# A plain root.
arbit_find_optix_sdk(dir version detail "" "${empty}/optix" "${WORK_DIR}/root")
expect("plain root" "${dir}" "${WORK_DIR}/root")

# A glob takes the highest version: 10.0.0 sorts above 9.1.0 naturally.
set(installs "${WORK_DIR}/installs")
fake_sdk("${installs}/NVIDIA-OptiX-SDK-9.1.0-linux64-x86_64" 90100)
fake_sdk("${installs}/NVIDIA-OptiX-SDK-10.0.0-linux64-x86_64" 100000)
arbit_find_optix_sdk(dir version detail "" "${installs}/NVIDIA-OptiX-SDK-*")
expect("glob takes the highest version" "${version}" "100000")

# A directory with optix.h but no stubs is not an SDK this build can use.
file(WRITE "${WORK_DIR}/partial/include/optix.h" "#define OPTIX_VERSION 80000\n")
arbit_find_optix_sdk(dir version detail "" "${WORK_DIR}/partial" "${WORK_DIR}/root")
expect("partial install is skipped" "${dir}" "${WORK_DIR}/root")

# Paths with spaces work, as on Windows.
fake_sdk("${WORK_DIR}/with space/OptiX SDK 9.0.0" 90000)
arbit_find_optix_sdk(dir version detail "" "${WORK_DIR}/with space/OptiX SDK *")
expect("root with spaces" "${dir}" "${WORK_DIR}/with space/OptiX SDK 9.0.0")

# The CUDA headers: the first root holding cuda.h wins, and none means empty, so
# an SDK without a toolkit counts as not found.
file(WRITE "${WORK_DIR}/toolkit/include/cuda.h" "")
file(WRITE "${WORK_DIR}/second/include/cuda.h" "")
arbit_find_cuda_headers(cuda "" "${empty}" "${WORK_DIR}/toolkit" "${WORK_DIR}/second")
expect("first root with cuda.h" "${cuda}" "${WORK_DIR}/toolkit/include")
arbit_find_cuda_headers(cuda "" "${empty}")
expect("no cuda.h" "${cuda}" "")

file(REMOVE_RECURSE "${WORK_DIR}")
message(STATUS "OptiX SDK discovery checks passed")
