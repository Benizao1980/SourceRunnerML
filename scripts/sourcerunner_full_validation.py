#!/usr/bin/env python3
"""
General SourceRunnerML source attribution, validation and prediction wrapper.

What this adds beyond the smoke-test SourceRunner run:
  1. restricts training to chosen source classes;
  2. compares multiple ML models by stratified cross-validation;
  3. performs fold-contained preprocessing and bootstrap/OOB validation;
  4. averages human predictions across independently trained bootstrap models;
  5. writes uncertainty summaries, per-class probabilities, CV reports, confusion matrices,
     and source-attribution summary tables.

This script expects SourceRunner-ready TSVs with a common genomic allele scheme.
The default CAMP locus prefix is retained for backwards compatibility.
"""

__version__ = "1.0.0"

import argparse
import json
import math
import os
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sourcerunner_features import AllelePreprocessor, select_shared_loci
from sourcerunner_tuning import fit_with_optional_tuning, load_tuning_config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SourceRunnerML multi-source allele-profile validation wrapper")
    p.add_argument("--train_file", required=True)
    p.add_argument("--predict_file", required=True)
    p.add_argument("--output_dir", default="sourcerunner_coli_cgmlst_full_outputs")
    p.add_argument("--run_name", default="coli_cgmlst_three_source_full")
    p.add_argument("--source_col", default="reduced")
    p.add_argument("--keep_sources", default="Poultry,Ruminant,Pig")
    p.add_argument("--loci_prefix", default="CAMP", help="Legacy default; overridden by --loci_pattern or --loci_file")
    p.add_argument("--loci_pattern", default="", help="Regex matched at beginning of locus column name")
    p.add_argument("--loci_file", default="", help="One locus name per line; requires exact columns in train and prediction tables")
    p.add_argument("--missingness", type=float, default=0.20)
    p.add_argument("--tuning_config", default="", help="Optional JSON model->parameter list grids; disabled by default")
    p.add_argument("--inner_cv_folds", type=int, default=3, help="Training-only folds for optional nested tuning")
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--models", default="random_forest,xgboost,logreg,lightgbm,catboost")
    p.add_argument("--selection_metric", default="balanced_accuracy", choices=["balanced_accuracy", "macro_f1", "weighted_f1", "accuracy"])
    p.add_argument("--bootstrap", type=int, default=100)
    p.add_argument("--burn_in", type=int, default=25)
    p.add_argument("--pred_bootstrap", type=int, default=50)
    p.add_argument("--min_confidence", type=float, default=0.60)
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--max_train_rows", type=int, default=0, help="Debug only: limit training rows after filtering")
    p.add_argument("--max_predict_rows", type=int, default=0, help="Debug only: limit prediction rows")
    return p.parse_args()


def timestamped_dir(base: str, run_name: str) -> Path:
    out = Path(base) / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def normalise_cell_values(df: pd.DataFrame, loci: List[str]) -> pd.DataFrame:
    # BIGSdb multi-allele cells are sometimes represented with punctuation/separators.
    # SourceRunner/ML should treat these as missing, not as numeric alleles.
    x = df[loci].copy()
    x = x.replace({"": np.nan, " ": np.nan, "-": np.nan, "NA": np.nan, "NaN": np.nan, "nan": np.nan})
    for c in loci:
        s = x[c].astype(str)
        bad = s.str.contains(r"[;,/|]", regex=True, na=False)
        if bad.any():
            x.loc[bad, c] = np.nan
    return x.apply(pd.to_numeric, errors="coerce")


def get_model(name: str, n_jobs: int, random_state: int):
    name = name.strip().lower()
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "logreg":
        return LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            class_weight="balanced",
            n_jobs=min(n_jobs, 4),
            random_state=random_state,
        )
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
            verbosity=0,
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            verbose=0,
            thread_count=n_jobs,
            random_seed=random_state,
        )
    raise ValueError(f"Unknown model: {name}")


def make_pipeline(model_name, cpus, seed, missingness):
    return Pipeline([
        ("alleles", AllelePreprocessor(missingness=missingness)),
        ("classifier", get_model(model_name, n_jobs=cpus, random_state=seed)),
    ])


def metric_dict(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def write_report_json_and_txt(outdir: Path, stem: str, payload: dict):
    (outdir / f"{stem}.json").write_text(json.dumps(payload, indent=2))
    lines = []
    def add(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                add(f"{prefix}{k}.", v)
        elif isinstance(obj, list):
            lines.append(f"{prefix[:-1]}: {', '.join(map(str, obj))}")
        else:
            lines.append(f"{prefix[:-1]}: {obj}")
    add("", payload)
    (outdir / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def cross_validate_model(model_name: str, X: pd.DataFrame, y: np.ndarray, labels: List[str], args: argparse.Namespace, outdir: Path, grids=None):
    start = time.time()
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_seed)
    y_pred = np.full(len(y), -1, dtype=int)
    fold_records = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        model = make_pipeline(model_name, args.cpus, args.random_seed, args.missingness)
        model, tuning = fit_with_optional_tuning(
            model, X.iloc[tr], y[tr], model_name=model_name,
            grids=grids or {}, metric=args.selection_metric,
            inner_folds=args.inner_cv_folds, seed=args.random_seed + fold,
        )
        y_pred[te] = model.predict(X.iloc[te])
        fold_records.append({
            "fold": fold,
            "train_n": len(tr),
            "test_n": len(te),
            "loci_retained": len(model.named_steps["alleles"].retained_loci_),
            "loci": ",".join(model.named_steps["alleles"].retained_loci_),
            "tuning": json.dumps(tuning, sort_keys=True),
        })
    if np.any(y_pred < 0):
        raise ValueError("Some CV isolates were not evaluated")
    pd.DataFrame(fold_records).to_csv(
        outdir / f"cv_fold_retained_loci__{model_name}.tsv", sep="\t", index=False
    )
    metrics = metric_dict(y, y_pred)
    cm = confusion_matrix(y, y_pred, labels=list(range(len(labels))))
    rep = classification_report(y, y_pred, labels=list(range(len(labels))),
                                target_names=labels, output_dict=True, zero_division=0)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        outdir / f"cv_confusion_matrix__{model_name}.tsv", sep="\t"
    )
    with open(outdir / f"cv_classification_report__{model_name}.json", "w") as fh:
        json.dump(rep, fh, indent=2)
    metrics["model"] = model_name
    metrics["seconds"] = round(time.time() - start, 2)
    metrics["status"] = "ok"
    return metrics

def stratified_bootstrap_indices(y: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    train_parts = []
    oob_mask = np.ones(len(y), dtype=bool)
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        sampled = rng.choice(idx, size=len(idx), replace=True)
        train_parts.append(sampled)
        # OOB relative to this class
        sampled_unique = np.unique(sampled)
        oob_mask[sampled_unique] = False
    train_idx = np.concatenate(train_parts)
    rng.shuffle(train_idx)
    oob_idx = np.where(oob_mask)[0]
    return train_idx, oob_idx


def fit_bootstrap_replicate(rep: int, model_name: str, X: pd.DataFrame, y: np.ndarray,
                            seed: int, cpus_per_model: int, missingness: float = 0.20,
                            grids=None, selection_metric="balanced_accuracy", inner_folds=3):
    rng = np.random.default_rng(seed + rep)
    tr, te = stratified_bootstrap_indices(y, rng)
    model = make_pipeline(model_name, cpus_per_model, seed + rep, missingness)
    model, tuning = fit_with_optional_tuning(
        model, X.iloc[tr], y[tr], model_name=model_name,
        grids=grids or {}, metric=selection_metric,
        inner_folds=inner_folds, seed=seed + rep,
    )
    if len(te) > 0:
        pred = model.predict(X.iloc[te])
        metrics = metric_dict(y[te], pred)
        metrics["oob_n"] = int(len(te))
    else:
        metrics = {"accuracy": np.nan, "balanced_accuracy": np.nan, "macro_f1": np.nan,
                   "weighted_f1": np.nan, "oob_n": 0}
    metrics["replicate"] = rep
    metrics["loci_retained"] = len(model.named_steps["alleles"].retained_loci_)
    metrics["tuning"] = json.dumps(tuning, sort_keys=True)
    return metrics, model

def entropy_from_probs(p: np.ndarray) -> np.ndarray:
    p2 = np.clip(p, 1e-12, 1.0)
    ent = -(p2 * np.log2(p2)).sum(axis=1)
    max_ent = math.log2(p.shape[1]) if p.shape[1] > 1 else 1.0
    return ent / max_ent


def main():
    args = parse_args()
    t0 = time.time()
    outdir = timestamped_dir(args.output_dir, args.run_name)
    print(f"Outputs -> {outdir}", flush=True)

    keep_sources = [x.strip() for x in args.keep_sources.split(",") if x.strip()]
    model_names = [x.strip() for x in args.models.split(",") if x.strip()]

    train = pd.read_csv(args.train_file, sep="\t", dtype=str, low_memory=False)
    pred = pd.read_csv(args.predict_file, sep="\t", dtype=str, low_memory=False)

    if args.source_col not in train.columns:
        raise SystemExit(f"Training file lacks source column: {args.source_col}")

    train = train[train[args.source_col].isin(keep_sources)].copy()
    if args.max_train_rows and len(train) > args.max_train_rows:
        # Sampling within each source must preserve the source-label column.
        # pandas >=3 no longer includes grouping columns in groupby.apply().
        n_groups = max(1, train[args.source_col].nunique())
        per_group_limit = max(1, args.max_train_rows // n_groups)
        train = pd.concat(
            [
                group.sample(
                    n=min(len(group), per_group_limit),
                    random_state=args.random_seed,
                )
                for _, group in train.groupby(args.source_col, sort=False)
            ],
            ignore_index=True,
        )
    if args.max_predict_rows and len(pred) > args.max_predict_rows:
        pred = pred.sample(args.max_predict_rows, random_state=args.random_seed).reset_index(drop=True)

    try:
        shared_loci, locus_method = select_shared_loci(
            train, pred, loci_file=args.loci_file,
            loci_pattern=args.loci_pattern, loci_prefix=args.loci_prefix,
        )
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    X_raw = normalise_cell_values(train, shared_loci)
    P_raw = normalise_cell_values(pred, shared_loci)
    # Do not fit locus filtering or the imputer here: every held-out fold
    # or bootstrap OOB replicate must have its own training-only fit.

    le = LabelEncoder()
    y = le.fit_transform(train[args.source_col].astype(str))
    labels = list(le.classes_)

    setup = {
        "train_file": args.train_file,
        "predict_file": args.predict_file,
        "source_col": args.source_col,
        "kept_sources": keep_sources,
        "training_rows": int(len(train)),
        "prediction_rows": int(len(pred)),
        "training_source_counts": {str(k): int(v) for k, v in train[args.source_col].value_counts().items()},
        "shared_loci_before_missingness_filter": int(len(shared_loci)),
        "loci_selected_before_missingness_filter": int(len(shared_loci)),
        "locus_selection_method": locus_method,
        "loci_file": args.loci_file,
        "loci_pattern": args.loci_pattern,
        "preprocessing": "train-partition-fitted CV and OOB; full training for final predictions",
        "loci_prefix": args.loci_prefix,
        "missingness_threshold": args.missingness,
        "models_requested": model_names,
        "cv_folds": args.cv_folds,
        "bootstrap_replicates": args.bootstrap,
        "burn_in": args.burn_in,
        "prediction_bootstrap_models": args.pred_bootstrap,
        "min_confidence": args.min_confidence,
        "selection_metric": args.selection_metric,
        "tuning_config": args.tuning_config,
    }
    (outdir / "shared_loci_selected.txt").write_text("\n".join(shared_loci) + "\n")
    write_report_json_and_txt(outdir, "run_setup", setup)
    train.to_csv(outdir / "training_filtered_used.tsv", sep="\t", index=False)

    if len(labels) < 2 or min(np.bincount(y)) < args.cv_folds:
        raise SystemExit(f"Each source requires at least {args.cv_folds} training isolates for CV")

    try:
        grids = load_tuning_config(
            args.tuning_config, model_names,
            lambda name: get_model(name, n_jobs=args.cpus, random_state=args.random_seed),
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Invalid tuning configuration: {exc}") from exc
    setup["tuning_grids"] = grids
    setup["inner_cv_folds"] = args.inner_cv_folds if grids else None
    write_report_json_and_txt(outdir, "run_setup", setup)

    # Model comparison by full out-of-fold CV.
    cv_rows = []
    for name in model_names:
        print(f"CV model: {name}", flush=True)
        try:
            cv_rows.append(cross_validate_model(name, X_raw, y, labels, args, outdir, grids=grids))
        except Exception as e:
            cv_rows.append({"model": name, "status": "failed", "error": f"{type(e).__name__}: {e}"})
            print(f"  skipped/failed: {e}", flush=True)
    cv_df = pd.DataFrame(cv_rows)
    cv_df.to_csv(outdir / "model_comparison_cv_summary.tsv", sep="\t", index=False)
    ok = cv_df[cv_df["status"].eq("ok")].copy()
    if ok.empty:
        raise SystemExit("All model comparisons failed")
    best_name = ok.sort_values(args.selection_metric, ascending=False).iloc[0]["model"]
    print(f"Best model by {args.selection_metric}: {best_name}", flush=True)

    # Proper bootstrap validation and prediction uncertainty using independently trained models.
    if args.burn_in >= args.bootstrap:
        raise SystemExit("burn_in must be less than bootstrap")
    cpus_per_model = max(1, args.cpus)
    print(f"Bootstrap fitting: {args.bootstrap} reps; sequential outer loop; cpus/model={cpus_per_model}", flush=True)
    results = []
    for i in range(args.bootstrap):
        if i == 0 or (i + 1) % 10 == 0 or (i + 1) == args.bootstrap:
            print(f"  bootstrap replicate {i + 1}/{args.bootstrap}", flush=True)
        results.append(fit_bootstrap_replicate(
            i, best_name, X_raw, y, args.random_seed, cpus_per_model,
            args.missingness, grids=grids,
            selection_metric=args.selection_metric, inner_folds=args.inner_cv_folds,
        ))
    boot_metrics = [m for m, _ in results]
    boot_models = [m for _, m in results]
    boot_df = pd.DataFrame(boot_metrics)
    boot_df.to_csv(outdir / "bootstrap_oob_metrics_all_replicates.tsv", sep="\t", index=False)
    post = boot_df[boot_df["replicate"] >= args.burn_in].copy()
    summary = post[["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "oob_n"]].agg(["mean", "std", "min", "median", "max"]).T
    summary.to_csv(outdir / "bootstrap_oob_metrics_post_burnin_summary.tsv", sep="\t")

    # Final prediction model may fit on all labelled training rows: no held-out scores are derived from it.
    final_pipeline = make_pipeline(best_name, args.cpus, args.random_seed, args.missingness)
    final_pipeline, final_tuning = fit_with_optional_tuning(
        final_pipeline, X_raw, y, model_name=best_name, grids=grids,
        metric=args.selection_metric, inner_folds=args.inner_cv_folds,
        seed=args.random_seed,
    )
    write_report_json_and_txt(outdir, "final_model_tuning", final_tuning)
    final_preprocessor = final_pipeline.named_steps["alleles"]
    loci = list(final_preprocessor.retained_loci_)
    (outdir / "retained_loci_final.txt").write_text("\n".join(loci) + "\n")
    setup["loci_retained"] = len(loci)
    write_report_json_and_txt(outdir, "run_setup", setup)
    with open(outdir / "final_model_fit_all_training.pkl", "wb") as fh:
        # Preserve old dictionary keys for consumers that transform 'loci' with 'imputer'.
        pickle.dump({"model": final_pipeline.named_steps["classifier"],
                     "label_encoder": le, "loci": loci,
                     "imputer": final_preprocessor.imputer_,
                     "args": vars(args),
                     "preprocessing_note": "Use loci in order and imputer.transform before model.predict_proba"}, fh)

    # Prediction across independent bootstrap models after burn-in.
    pred_models = boot_models[args.burn_in:args.burn_in + args.pred_bootstrap]
    if len(pred_models) == 0:
        raise SystemExit("No bootstrap models available for prediction")
    print(f"Predicting with {len(pred_models)} post-burn-in bootstrap models", flush=True)
    prob_stack = []
    call_stack = []
    for m in pred_models:
        probs = m.predict_proba(P_raw)
        # Align probability columns to encoded class order if needed.
        aligned = np.zeros((len(P_raw), len(labels)), dtype=float)
        for j, cls in enumerate(m.classes_):
            aligned[:, int(cls)] = probs[:, j]
        prob_stack.append(aligned)
        call_stack.append(np.argmax(aligned, axis=1))
    prob_stack = np.stack(prob_stack, axis=0)
    call_stack = np.stack(call_stack, axis=0)
    mean_probs = prob_stack.mean(axis=0)
    sd_probs = prob_stack.std(axis=0)
    pred_num = mean_probs.argmax(axis=1)
    pred_source = le.inverse_transform(pred_num)
    max_prob = mean_probs.max(axis=1)
    max_prob_sd = sd_probs[np.arange(len(P_raw)), pred_num]
    consensus = np.array([(call_stack[:, i] == pred_num[i]).mean() for i in range(len(P_raw))])
    uncertainty_entropy = entropy_from_probs(mean_probs)
    pred_filtered = np.where(max_prob < args.min_confidence, "Uncertain", pred_source)

    out_pred = pred.copy()
    out_pred["predicted_source"] = pred_source
    out_pred["predicted_filtered"] = pred_filtered
    out_pred["max_probability"] = max_prob
    out_pred["max_probability_sd_across_bootstrap_models"] = max_prob_sd
    out_pred["bootstrap_consensus"] = consensus
    out_pred["normalised_entropy"] = uncertainty_entropy
    for j, lab in enumerate(labels):
        out_pred[f"prob_mean_{lab}"] = mean_probs[:, j]
        out_pred[f"prob_sd_{lab}"] = sd_probs[:, j]
    out_pred.to_csv(outdir / "human_predictions_bootstrap_ensemble.tsv", sep="\t", index=False)

    prob_df = pd.DataFrame(mean_probs, columns=[f"prob_mean_{x}" for x in labels])
    sd_df = pd.DataFrame(sd_probs, columns=[f"prob_sd_{x}" for x in labels])
    prob_df.insert(0, "row_index", np.arange(len(prob_df)))
    prob_df.to_csv(outdir / "prediction_probability_means.tsv", sep="\t", index=False)
    sd_df.insert(0, "row_index", np.arange(len(sd_df)))
    sd_df.to_csv(outdir / "prediction_probability_sds.tsv", sep="\t", index=False)

    source_summary = out_pred["predicted_filtered"].value_counts(dropna=False).rename_axis("predicted_filtered").reset_index(name="count")
    source_summary["percent"] = source_summary["count"] / len(out_pred) * 100
    source_summary.to_csv(outdir / "source_attribution_summary_filtered.tsv", sep="\t", index=False)

    raw_summary = out_pred["predicted_source"].value_counts(dropna=False).rename_axis("predicted_source").reset_index(name="count")
    raw_summary["percent"] = raw_summary["count"] / len(out_pred) * 100
    raw_summary.to_csv(outdir / "source_attribution_summary_raw.tsv", sep="\t", index=False)

    if "clonal_complex" in out_pred.columns:
        cc_summary = pd.crosstab(out_pred["clonal_complex"], out_pred["predicted_filtered"])
        cc_summary.to_csv(outdir / "source_by_clonal_complex_filtered.tsv", sep="\t")
        cc_raw = pd.crosstab(out_pred["clonal_complex"], out_pred["predicted_source"])
        cc_raw.to_csv(outdir / "source_by_clonal_complex_raw.tsv", sep="\t")

    uncertainty_summary = {
        "best_model": best_name,
        "prediction_models_used": len(pred_models),
        "mean_max_probability": float(np.mean(max_prob)),
        "median_max_probability": float(np.median(max_prob)),
        "mean_max_probability_sd": float(np.mean(max_prob_sd)),
        "median_max_probability_sd": float(np.median(max_prob_sd)),
        "mean_bootstrap_consensus": float(np.mean(consensus)),
        "median_bootstrap_consensus": float(np.median(consensus)),
        "mean_normalised_entropy": float(np.mean(uncertainty_entropy)),
        "uncertain_count": int((out_pred["predicted_filtered"] == "Uncertain").sum()),
        "uncertain_percent": float((out_pred["predicted_filtered"] == "Uncertain").mean() * 100),
        "runtime_seconds": round(time.time() - t0, 2),
    }
    write_report_json_and_txt(outdir, "prediction_uncertainty_summary", uncertainty_summary)

    print("Done", flush=True)
    print(json.dumps(uncertainty_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
