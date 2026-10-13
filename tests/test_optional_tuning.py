"""Regression checks for the optional, fold-local hyperparameter tuning path."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sourcerunner_full_validation import make_pipeline, get_model
from sourcerunner_tuning import load_tuning_config, fit_with_optional_tuning


class TuningTests(unittest.TestCase):
    def test_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "grid.json"
            factory = lambda name: get_model(name, n_jobs=1, random_state=42)
            p.write_text(json.dumps({"random_forest": {"max_depth": [None, 4]}}))
            self.assertIn("random_forest", load_tuning_config(p, ["random_forest"], factory))
            p.write_text(json.dumps({"random_forest": {"not_a_parameter": [3]}}))
            with self.assertRaisesRegex(ValueError, "unknown"):
                load_tuning_config(p, ["random_forest"], factory)
            p.write_text(json.dumps({"random_forest": {"n_jobs": [2]}}))
            with self.assertRaisesRegex(ValueError, "prohibited"):
                load_tuning_config(p, ["random_forest"], factory)
            p.write_text(json.dumps({"random_forest": {"n_estimators": []}}))
            with self.assertRaisesRegex(ValueError, "nonempty"):
                load_tuning_config(p, ["random_forest"], factory)

    def test_small_tuning_on_partition_only(self):
        n = 30
        X = pd.DataFrame({"A": [i % 4 for i in range(n)],
                          "B": [np.nan if i % 6 == 0 else i % 3 for i in range(n)]})
        y = np.asarray([i % 2 for i in range(n)])
        model = make_pipeline("random_forest", 1, 42, 0.5)
        fitted, detail = fit_with_optional_tuning(
            model, X.iloc[:24], y[:24], model_name="random_forest",
            grids={"random_forest": {"n_estimators": [8, 12]}},
            metric="balanced_accuracy", inner_folds=2, seed=42,
        )
        self.assertTrue(detail["tuned"])
        self.assertIn(detail["best_params"]["n_estimators"], [8, 12])
        self.assertEqual(detail["candidates"], 2)
        self.assertEqual(len(fitted.predict(X.iloc[24:])), 6)

    def test_default_model_not_tuned(self):
        X = pd.DataFrame({"A": [1, 1, 2, 2, 1, 2]})
        y = np.array([0, 0, 1, 1, 0, 1])
        fitted, detail = fit_with_optional_tuning(
            make_pipeline("random_forest", 1, 42, 0.5),
            X, y, model_name="random_forest", grids={},
            metric="balanced_accuracy", inner_folds=2, seed=42,
        )
        self.assertFalse(detail["tuned"])
        self.assertEqual(detail["best_params"], {})
        self.assertEqual(len(fitted.predict(X)), 6)


if __name__ == "__main__":
    unittest.main()
