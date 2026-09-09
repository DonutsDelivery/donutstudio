#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def run_export_visual_plan_session_test(root: Path) -> None:
    sources = [
        root / "tests/export_visual_plan_session_tests.cpp",
        root / "src/export_visual_plan_session.cpp",
        root / "src/exporter_visual_plan_callbacks.cpp",
        root / "src/visual_plan_execution_telemetry.cpp",
        root / "src/visual_renderer_telemetry.cpp",
    ]
    compiler = os.environ.get("CXX", "c++")
    lock = shutil.which("host-heavy-build")
    if lock is None:
        local_lock = Path.home() / ".local/bin/host-heavy-build"
        lock = str(local_lock) if local_lock.is_file() else None
    with tempfile.TemporaryDirectory(prefix="export-visual-plan-session-") as directory:
        for have_onnx in (0, 1):
            executable = Path(directory) / f"test-{have_onnx}"
            command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                       "-pthread", f"-DARBIT_HAVE_ONNX={have_onnx}",
                       "-I", str(root / "src"), *(str(source) for source in sources),
                       "-o", str(executable)]
            if lock is not None:
                command = [lock, "run", "--project", "DonutStudio-telemetry-contract",
                           "--worktree", str(root), "--wait", "21600", "--", *command]
            subprocess.run(command, check=True)
            subprocess.run([str(executable)], check=True)
