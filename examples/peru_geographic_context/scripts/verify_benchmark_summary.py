#!/usr/bin/env python3
"""Check arithmetic of the documented aggregate Peru C. jejuni benchmark.

This checks frozen SUMMARY COUNTS only; it does not rerun models or access genomes.
Uses Python standard library only.
"""
import csv
from pathlib import Path

FILE = Path(__file__).resolve().parents[1] / "results" / "cjejuni_known_source_benchmark.tsv"


def main():
    with FILE.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 3, f"Expected three reference strategies, found {len(rows)}"
    for row in rows:
        n_p = int(row["poultry_total"])
        ok_p = int(row["poultry_correct"])
        wrong_p = int(row["poultry_to_ruminant"])
        n_r = int(row["ruminant_total"])
        ok_r = int(row["ruminant_correct"])
        wrong_r = int(row["ruminant_to_poultry"])
        assert n_p == 1076 and n_r == 149
        assert ok_p + wrong_p == n_p
        assert ok_r + wrong_r == n_r
        ba = ((ok_p / n_p) + (ok_r / n_r)) / 2
        # The README records three-decimal rounded BA.
        assert abs(ba - float(row["balanced_accuracy_reported"])) < 0.00055
        print(f"{row['strategy']}: n={n_p+n_r}, balanced accuracy from counts={ba:.3f}")
    print("Aggregate arithmetic checks passed. This is NOT an analysis reproduction.")


if __name__ == "__main__":
    main()
