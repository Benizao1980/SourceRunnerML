#!/usr/bin/env python3
"""Minimal import smoke test for SourceRunnerML scripts."""
import importlib.util
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FILES = [
    "SourceRunnerML.py",
    "utils_v1_0.py",
    "sourcerunner_full_validation.py",
    "sourcerunner_prediction_postprocess.py",
    "source_runner_preflight.py",
]

ok = True
for fn in FILES:
    path = ROOT / fn
    if not path.exists():
        print("MISSING", fn)
        ok = False
        continue
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        print("OK", fn)
    except Exception as e:
        print("ERROR", fn, repr(e))
        ok = False

sys.exit(0 if ok else 1)
