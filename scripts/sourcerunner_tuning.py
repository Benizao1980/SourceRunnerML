"""Optional, leakage-aware hyperparameter grid searches for SourceRunnerML.

The default pipeline never tunes; this module is only used when explicitly requested.
GridSearchCV refits preprocessing within each INNER fold. The caller must provide
only its current OUTER training fold or bootstrap in-bag rows.
"""
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold

FORBIDDEN = {"random_state", "n_jobs", "thread_count", "nthread", "verbose", "verbosity"}
METRIC_NAMES = {
    "balanced_accuracy": "balanced_accuracy",
    "macro_f1": "f1_macro",
    "weighted_f1": "f1_weighted",
    "accuracy": "accuracy",
}


def load_tuning_config(path, requested_models, factory):
    """Validate a JSON model->parameter-grid mapping against estimator parameters."""
    if not path:
        return {}
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Tuning configuration must be a nonempty JSON object")
    grids = {}
    for model_name, grid in raw.items():
        if model_name not in requested_models:
            raise ValueError(f"Tuning configuration refers to unrequested model: {model_name}")
        if not isinstance(grid, dict) or not grid:
            raise ValueError(f"{model_name}: tuning grid must be nonempty parameter dictionary")
        supported = set(factory(model_name).get_params())
        n_candidates = 1
        for key, values in grid.items():
            if key not in supported:
                raise ValueError(f"{model_name}: unknown estimator hyperparameter: {key}")
            if key in FORBIDDEN:
                raise ValueError(f"{model_name}: tuning {key} is prohibited for reproducibility/resource safety")
            if not isinstance(values, list) or not values:
                raise ValueError(f"{model_name}.{key}: supply a nonempty list of candidate values")
            if any(not (v is None or isinstance(v, (int, float, bool, str))) for v in values):
                raise ValueError(f"{model_name}.{key}: candidates must be scalar JSON values")
            n_candidates *= len(values)
        if n_candidates > 64:
            raise ValueError(f"{model_name}: grid has {n_candidates} candidates (maximum 64)")
        grids[model_name] = grid
    return grids


def fit_with_optional_tuning(pipeline, X, y, *, model_name, grids, metric,
                             inner_folds, seed):
    """Return fitted Pipeline and auditable tuning metadata.

    The caller has already withheld the evaluation rows. With tuning enabled,
    each candidate is evaluated by inner CV *only* on these training rows.
    """
    grid = grids.get(model_name)
    if not grid:
        pipeline.fit(X, y)
        return pipeline, {"tuned": False, "best_params": {},
                          "inner_cv_score": None, "candidates": 0}
    counts = np.bincount(np.asarray(y, dtype=int))
    counts = counts[counts > 0]
    if not len(counts) or min(counts) < 2:
        raise ValueError(f"{model_name}: fewer than two training examples in some class for inner CV")
    k = min(int(inner_folds), int(min(counts)))
    if k < 2:
        raise ValueError("inner_folds must be at least 2")
    estimator_grid = {"classifier__" + key: values for key, values in grid.items()}
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=estimator_grid,
        scoring=METRIC_NAMES[metric],
        cv=StratifiedKFold(n_splits=k, shuffle=True, random_state=seed),
        n_jobs=1,  # avoid inner/outer oversubscription; classifier may use its own CPU limit
        refit=True,
        error_score="raise",
    )
    search.fit(X, y)
    best = {key.removeprefix("classifier__"): value
            for key, value in search.best_params_.items()}
    return search.best_estimator_, {
        "tuned": True,
        "best_params": best,
        "inner_folds": k,
        "inner_cv_score": float(search.best_score_),
        "candidates": len(search.cv_results_["params"]),
    }
