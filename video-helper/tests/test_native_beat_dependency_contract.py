#!/usr/bin/env python3
"""Source-contract checks; runtime/audio quality is covered separately."""
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class NativeBeatDependencyContract(unittest.TestCase):
    def test_helper_has_no_aubio_build_or_runtime_calls(self):
        for path in ("video-helper/CMakeLists.txt", "video-helper/src/media.cpp",
                     "video-helper/src/media.h", "video-helper/src/main.cpp",
                     "video-helper/src/native_beat_detector.h"):
            text = (ROOT / path).read_text()
            self.assertNotRegex(text, r"(?i)aubio", path)

    def test_platform_install_lists_do_not_install_aubio(self):
        for path in ("Dockerfile.linux-videohelper", ".github/workflows/build-linux.yml",
                     ".github/workflows/build-windows.yml", ".github/workflows/build-macos.yml",
                     "scripts/build-windows-videohelper-docker.sh"):
            text = (ROOT / path).read_text()
            self.assertNotRegex(text, r"libaubio-dev|mingw-w64-x86_64-aubio", path)
            self.assertNotRegex(text, r"(?:brew install|ffmpeg)\s+aubio", path)

    def test_rpc_retains_fields_and_declares_fixed_grid(self):
        text = (ROOT / "video-helper/src/main.cpp").read_text()
        response = text.split('if (method == "detect_beats")', 1)[1].split(
            'if (method == "stabilize_detect")', 1)[0]
        for field in ("bpm", "confidence", "beats"):
            self.assertIn('"' + field + '"', response)
        self.assertIn('{ "engine", "donutstudio-native" }', response)
        self.assertIn('{ "timingModel", "constant-tempo" }', response)

    def test_capability_is_available_and_describes_backend(self):
        text = (ROOT / "video-helper/src/main.cpp").read_text()
        version = text.split('if (method == "version")', 1)[1].split(
            'if (method == "model_payload_begin")', 1)[0]
        self.assertIn('{ "beatDetection", true }', version)
        self.assertIn('{ "beatDetectionEngine", "donutstudio-native" }', version)
        self.assertIn('{ "beatDetectionTimingModel", "constant-tempo" }', version)

    def test_owned_detector_include_resolves_without_juce(self):
        header = ROOT / "video-helper/src/native_beat_detector.h"
        include = re.search(r'#include "([^"]+)"', header.read_text()).group(1)
        detector = (header.parent / include).resolve()
        self.assertEqual(detector, ROOT / "shared/ArbitBpmDetector.h")
        self.assertTrue(detector.is_file())
        self.assertNotRegex(detector.read_text(), r'#include\s*[<"](?:juce|aubio)')

    def test_old_plugin_include_forwards_to_shared_source(self):
        wrapper = (ROOT / "plugin/Source/ArbitBpmDetector.h").read_text()
        self.assertIn('#include "../../shared/ArbitBpmDetector.h"', wrapper)

    def test_windows_staging_clears_then_follows_imports(self):
        script = (ROOT / "scripts/build-windows-videohelper-docker.sh").read_text()
        self.assertIn('rm -rf "$STAGE"; mkdir -p "$STAGE"', script)
        self.assertIn('collect "$STAGE/arbit-video-helper.exe"', script)
        self.assertIn('collect "$STAGE/$dll"', script)

    def test_leaf_test_registered(self):
        cmake = (ROOT / "video-helper/CMakeLists.txt").read_text()
        self.assertIn("add_executable(arbit-native-beat-detector-tests", cmake)
        self.assertIn("add_test(NAME native_beat_detector COMMAND arbit-native-beat-detector-tests)", cmake)


if __name__ == "__main__":
    unittest.main()
