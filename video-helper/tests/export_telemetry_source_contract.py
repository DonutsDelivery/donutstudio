#!/usr/bin/env python3
from pathlib import Path
import sys

from export_visual_plan_session_test_runner import run_export_visual_plan_session_test

root = Path(sys.argv[1])
run_export_visual_plan_session_test(root)
print("export telemetry semantic contract: PASS")
