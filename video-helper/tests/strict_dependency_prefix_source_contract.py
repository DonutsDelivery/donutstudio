#!/usr/bin/env python3
"""Guard opt-in discovery and helper-only export wiring without compiling."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SourceContract(unittest.TestCase):
    def test_default_libav_discovery_is_retained(self):
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('if(ARBIT_VIDEO_DEPENDENCY_PREFIX)\n    arbit_find_prefix_av()\nelse()\n'
                      '    pkg_check_modules(AV REQUIRED IMPORTED_TARGET', cmake)
        for dependency in ('LUA', 'GLFW3', 'NCNN', 'MVK'):
            self.assertNotIn('pkg_check_modules(' + dependency,
                             (ROOT / 'cmake/StrictDependencyPrefix.cmake').read_text())

    def test_pkg_config_environment_is_restored_and_cross_root_is_bypassed(self):
        module = (ROOT / 'cmake/StrictDependencyPrefix.cmake').read_text()
        self.assertIn('NO_CMAKE_PATH NO_CMAKE_ENVIRONMENT_PATH', module)
        self.assertIn('NO_DEFAULT_PATH NO_CMAKE_FIND_ROOT_PATH', module)
        self.assertIn('set(ENV{PKG_CONFIG_PATH} "")', module)
        self.assertIn('set(ENV{PKG_CONFIG_SYSROOT_DIR} "")', module)
        self.assertIn('set(ENV{${_name}} "${_saved_${_name}}")', module)
        self.assertIn('unset(ENV{${_name}})', module)
        self.assertIn('--binding ${_binding_args}', module)
        self.assertIn('ARBIT_PREFIX_AV_CFLAGS_OTHER OR ARBIT_PREFIX_AV_LDFLAGS_OTHER', module)

    def test_audited_av_headers_precede_imported_ambient_roots_and_have_real_input_guard(self):
        module = (ROOT / 'cmake/StrictDependencyPrefix.cmake').read_text()
        self.assertIn('add_library(arbit-prefix-av INTERFACE)', module)
        self.assertIn('add_library(PkgConfig::AV ALIAS arbit-prefix-av)', module)
        self.assertNotIn('add_library(PkgConfig::AV INTERFACE IMPORTED', module)
        self.assertIn('set(CMAKE_EXPORT_COMPILE_COMMANDS ON PARENT_SCOPE)', module)
        self.assertIn('cmake_language(DEFER CALL arbit_check_prefix_av_consumers)', module)
        self.assertIn('if("PkgConfig::AV" IN_LIST _links)', module)
        self.assertIn('add_dependencies("${_target}" arbit-prefix-av-header-check)', module)
        self.assertIn('--compile-build "${CMAKE_CURRENT_BINARY_DIR}"', module)
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('cmake_minimum_required(VERSION 3.22)', cmake)
        self.assertIn('link_libraries(nlohmann_json::nlohmann_json)', cmake)

    def test_export_adapter_is_private_and_opt_in(self):
        cmake = (ROOT / 'CMakeLists.txt').read_text()
        guarded = cmake.split('include(cmake/Kvazaar10.cmake)', 1)[1].split('add_test(', 1)[0]
        self.assertIn('if(ARBIT_BUILD_KVAZAAR10_ADAPTER)', guarded)
        self.assertIn('if(NOT TARGET arbit-kvazaar10-adapter)', guarded)
        self.assertIn('target_link_libraries(arbit-video-helper PRIVATE arbit-kvazaar10-adapter)', guarded)
        self.assertIn('target_compile_definitions(arbit-video-helper PRIVATE ARBIT_HAVE_KVAZAAR10_EXPORT=1)', guarded)
        self.assertNotIn('KVZ_BIT_DEPTH', cmake)
        self.assertNotIn('kvazaar10_source_SOURCE_DIR', cmake)
        adapter = (ROOT / 'cmake/Kvazaar10.cmake').read_text()
        self.assertIn('option(ARBIT_BUILD_KVAZAAR10_ADAPTER "Build the isolated helper ten-bit HEVC adapter" OFF)', adapter)
        self.assertIn('if(NOT ARBIT_BUILD_KVAZAAR10_ADAPTER)\n    return()\nendif()', adapter)
        apple = adapter.split('if(APPLE)', 1)[1].split('elseif(MSVC', 1)[0]
        self.assertIn('NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Darwin"', apple)
        self.assertIn('NOT CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "^(arm64|aarch64)$"', apple)
        self.assertIn('NOT CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"', apple)
        self.assertIn('foreach(_tool NM OBJCOPY RANLIB)', apple)
        self.assertIn('CACHE FILEPATH "Explicit recorded LLVM ${_tool} executable"', apple)
        self.assertIn('--format macho-arm64', apple)
        for flag, tool in (('--nm', 'NM'), ('--objcopy', 'OBJCOPY'), ('--ranlib', 'RANLIB')):
            self.assertIn(flag + ' "${ARBIT_KVAZAAR10_LLVM_' + tool + '}"', apple)
        self.assertIn('--llvm-version "${ARBIT_KVAZAAR10_LLVM_VERSION}"', apple)
        self.assertIn('${_kvz10_namespace_args} --check-tools', apple)
        self.assertIn('if(NOT _kvz10_tool_result EQUAL 0)\n        message(FATAL_ERROR', apple)
        self.assertIn('elseif(MSVC OR NOT CMAKE_NM OR NOT CMAKE_OBJCOPY OR NOT CMAKE_RANLIB)\n'
                      '    message(FATAL_ERROR', adapter)
        self.assertIn('target_compile_definitions(arbit-kvazaar10-adapter PRIVATE KVZ_BIT_DEPTH=10', adapter)


if __name__ == '__main__':
    unittest.main()
