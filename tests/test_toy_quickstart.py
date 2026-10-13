#!/usr/bin/env python3
"""Dependency-free consistency test for tutorial data generation."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "examples" / "toy_quickstart" / "make_toy_data.py"


class ToyGeneratorTest(unittest.TestCase):
    def test_reproducible_toy_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a"
            b = Path(tmp) / "b"
            for dest in (a, b):
                subprocess.run(
                    [sys.executable, str(GENERATOR), "--outdir", str(dest), "--seed", "13"],
                    check=True, capture_output=True, text=True,
                )
            self.assertEqual((a / "train.tsv").read_bytes(), (b / "train.tsv").read_bytes())
            self.assertEqual((a / "predict.tsv").read_bytes(), (b / "predict.tsv").read_bytes())
            with (a / "train.tsv").open(newline="") as handle:
                train = list(csv.DictReader(handle, delimiter="\t"))
            with (a / "predict.tsv").open(newline="") as handle:
                predict = list(csv.DictReader(handle, delimiter="\t"))
            loci = [key for key in train[0] if key.startswith("CAMP")]
            self.assertEqual(len(train), 180)
            self.assertEqual(len(predict), 30)
            self.assertEqual(len(loci), 120)
            self.assertEqual({v: sum(r["source"] == v for r in train) for v in ["Poultry", "Ruminant", "Pig"]},
                             {"Poultry": 60, "Ruminant": 60, "Pig": 60})
            self.assertEqual(set(loci), {k for k in predict[0] if k.startswith("CAMP")})
            self.assertEqual(len({r["id"] for r in train}), len(train))
            self.assertEqual(len({r["id"] for r in predict}), len(predict))
            self.assertTrue(all((not r[k]) or r[k].isdigit() for r in train for k in loci))


if __name__ == "__main__":
    unittest.main()
