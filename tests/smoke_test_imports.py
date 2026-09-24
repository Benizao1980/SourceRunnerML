#!/usr/bin/env python3
"""Minimal import smoke test for SourceRunnerML scripts."""

import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

FILES = [
    "SourceRunnerML.py",
    "utils_v1_0.py",
    "sourcerunner_full_validation.py",
    "sourcerunner_prediction_postprocess.py",
    "source_runner_preflight.py",
]

ok = True
for fn in FILES:
    path = SCRIPTS / fn
    if not path.exists():
        print("MISSING", path.relative_to(ROOT))
        ok = False
        continue

    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("OK", path.relative_to(ROOT))
    except Exception as e:
        print("ERROR", path.relative_to(ROOT), repr(e))
        ok = False

sys.exit(0 if ok else 1)
