#!/usr/bin/env python3
from pathlib import Path

from export_visual_plan_session_test_runner import run_export_visual_plan_session_test

root = Path(__file__).resolve().parents[1]
run_export_visual_plan_session_test(root)
print("generator/particle telemetry semantic contract: PASS")
