"""No-data regression tests for partition-safe genomic features and locus selection."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sourcerunner_features import AllelePreprocessor, select_shared_loci


class FeatureTests(unittest.TestCase):
    def test_held_out_rows_do_not_change_imputation_or_locus_filtering(self):
        # Only training rows may determine the modal allele and locus missingness.
        train = pd.DataFrame({
            "LocusA": [5.0, 5.0, np.nan, 5.0],
            "LocusB": [np.nan, np.nan, 8.0, np.nan],
            "LocusC": [3.0, 3.0, 3.0, 3.0],
        })
        held_out = pd.DataFrame({
            "LocusA": [999.0, np.nan, np.nan],
            "LocusB": [8.0, 8.0, 8.0],
            "LocusC": [4.0, 4.0, 4.0],
        })
        step = AllelePreprocessor(missingness=0.50).fit(train)
        self.assertEqual(step.retained_loci_, ["LocusA", "LocusC"])
        result = step.transform(held_out)
        self.assertEqual(result.shape, (3, 2))
        self.assertEqual(result[1, 0], 5.0)
        self.assertEqual(result[0, 1], 4.0)
        # Refit with held-out would incorrectly retain B (4 of 7 nonmissing).
        wrong = AllelePreprocessor(missingness=0.50).fit(pd.concat([train, held_out]))
        self.assertIn("LocusB", wrong.retained_loci_)
        self.assertNotIn("LocusB", step.retained_loci_)

    def test_allele_ids_are_unchanged_nominal_values(self):
        step = AllelePreprocessor(missingness=1.0).fit(
            pd.DataFrame({"A": [10.0, 1000.0], "B": [3.0, 3.0]})
        )
        self.assertEqual(step.transform(pd.DataFrame({"A": [1000.0], "B": [3.0]}))[0, 0], 1000.0)

    def test_selection_file_regex_prefix_and_absent_loci(self):
        train = pd.DataFrame(columns=["id", "source", "SAL_A", "abc2", "SAL_C"])
        pred = pd.DataFrame(columns=["id", "SAL_A", "abc2", "SAL_C"])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "loci.tsv"
            p.write_text("# organism-specific allele scheme\nlocus\nabc2\nSAL_C\n")
            self.assertEqual(select_shared_loci(train, pred, loci_file=p), (["abc2", "SAL_C"], "file"))
            self.assertEqual(select_shared_loci(train, pred, loci_pattern="^SAL_"), (["SAL_A", "SAL_C"], "regex"))
            self.assertEqual(select_shared_loci(train, pred, loci_prefix="SAL_"), (["SAL_A", "SAL_C"], "prefix"))
            p.write_text("SAL_C\nabsent_locus\n")
            with self.assertRaisesRegex(ValueError, "absent"):
                select_shared_loci(train, pred, loci_file=p)
            p.write_text("SAL_C\nSAL_C\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                select_shared_loci(train, pred, loci_file=p)

    def test_all_missing_loci_not_retained(self):
        with self.assertRaisesRegex(ValueError, "No loci retained"):
            AllelePreprocessor(missingness=1.0).fit(pd.DataFrame({"A": [np.nan, np.nan]}))


if __name__ == "__main__":
    unittest.main()
