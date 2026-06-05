#!/usr/bin/env python3
"""
Preflight checker and normaliser for SourceRunnerML input files.

Checks:
  - delimiter/readability
  - required metadata/source columns
  - required locus columns / regex-selected loci
  - shared columns between training and prediction files
  - source-label distribution and classes below minimum size
  - missing/non-numeric allele calls
  - duplicate sample IDs

Optionally writes clean TSVs for SourceRunnerML, because SourceRunnerML currently
expects tab-separated files.
"""
__version__ = "1.0.0"

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

DEFAULT_MLST_LOCI = ["aspA", "glnA", "gltA", "glyA", "pgm", "tkt", "uncA"]


def read_table(path: Path) -> Tuple[pd.DataFrame, str]:
    """Read CSV/TSV with simple delimiter inference."""
    # Try pandas sniffing first
    try:
        df = pd.read_csv(path, sep=None, engine="python", dtype=str)
        # crude delimiter label for report
        first = path.read_text(errors="replace").splitlines()[0]
        delim = "tab" if "\t" in first and first.count("\t") >= first.count(",") else "comma"
        return df, delim
    except Exception:
        # Fall back to TSV then CSV
        for sep, label in [("\t", "tab"), (",", "comma")]:
            try:
                return pd.read_csv(path, sep=sep, dtype=str, low_memory=False), label
            except Exception:
                pass
    raise RuntimeError(f"Could not read {path}")


def select_loci(df: pd.DataFrame, loci: str, loci_pattern: str) -> List[str]:
    if loci:
        wanted = [x.strip() for x in loci.split(",") if x.strip()]
        return [x for x in wanted if x in df.columns]
    if loci_pattern:
        rgx = re.compile(loci_pattern)
        return [c for c in df.columns if rgx.match(c)]
    return [c for c in DEFAULT_MLST_LOCI if c in df.columns]


def summarise_file(df: pd.DataFrame, name: str, source_col: str, id_col: str, loci: List[str], min_source_size: int):
    out = {
        "name": name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "has_source_col": source_col in df.columns,
        "has_id_col": id_col in df.columns,
        "n_loci": len(loci),
        "loci": loci,
        "duplicate_ids": None,
        "source_counts": None,
        "classes_below_min_source_size": None,
        "missingness_by_locus": {},
        "non_numeric_by_locus": {},
    }
    if id_col in df.columns:
        out["duplicate_ids"] = int(df[id_col].duplicated().sum())
    if source_col in df.columns:
        counts = df[source_col].fillna("<NA>").value_counts(dropna=False)
        out["source_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["classes_below_min_source_size"] = {str(k): int(v) for k, v in counts[counts < min_source_size].items()}
    for locus in loci:
        s = df[locus]
        coerced = pd.to_numeric(s, errors="coerce")
        missing_orig = s.isna() | s.astype(str).str.upper().isin(["NA", "NAN", "", "NONE", "NULL"])
        out["missingness_by_locus"][locus] = float(missing_orig.mean())
        out["non_numeric_by_locus"][locus] = int((coerced.isna() & ~missing_orig).sum())
    return out


def write_tsv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def main():
    ap = argparse.ArgumentParser(description="Check and normalise SourceRunnerML inputs")
    ap.add_argument("--train", required=True, help="Training CSV/TSV; here this is usually non-human/source data")
    ap.add_argument("--predict", required=True, help="Prediction CSV/TSV; here this is usually human data")
    ap.add_argument("--source_col", default="reduced")
    ap.add_argument("--id_col", default="id")
    ap.add_argument("--loci", default=",".join(DEFAULT_MLST_LOCI), help="Comma-separated locus columns")
    ap.add_argument("--loci_pattern", default="", help="Regex for locus columns; overrides --loci if supplied")
    ap.add_argument("--min_source_size", type=int, default=100)
    ap.add_argument("--exclude_sources", default="Missing", help="Comma-separated source labels to remove from TRAINING only")
    ap.add_argument("--outdir", default="sourcerunner_preflight")
    ap.add_argument("--write_tsv", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    train_path, pred_path = Path(args.train), Path(args.predict)

    train, train_delim = read_table(train_path)
    pred, pred_delim = read_table(pred_path)

    exclude = [x.strip() for x in args.exclude_sources.split(",") if x.strip()]
    if args.source_col in train.columns and exclude:
        train_clean = train[~train[args.source_col].isin(exclude)].copy()
    else:
        train_clean = train.copy()

    loci_train = select_loci(train_clean, args.loci if not args.loci_pattern else "", args.loci_pattern)
    loci_pred = select_loci(pred, args.loci if not args.loci_pattern else "", args.loci_pattern)
    shared_loci = [l for l in loci_train if l in loci_pred]

    report = {
        "inputs": {
            "train_path": str(train_path),
            "predict_path": str(pred_path),
            "train_detected_delimiter": train_delim,
            "predict_detected_delimiter": pred_delim,
        },
        "settings": vars(args),
        "excluded_training_sources": exclude,
        "shared_columns": int(len(set(train_clean.columns).intersection(pred.columns))),
        "columns_only_in_train": sorted(set(train_clean.columns) - set(pred.columns)),
        "columns_only_in_predict": sorted(set(pred.columns) - set(train_clean.columns)),
        "loci_shared": shared_loci,
        "train_summary_after_exclusions": summarise_file(train_clean, "train", args.source_col, args.id_col, shared_loci, args.min_source_size),
        "predict_summary": summarise_file(pred, "predict", args.source_col, args.id_col, shared_loci, args.min_source_size),
        "status": "PASS",
        "problems": [],
        "warnings": [],
    }

    if args.source_col not in train_clean.columns:
        report["problems"].append(f"Training file lacks source column: {args.source_col}")
    if args.id_col not in train_clean.columns:
        report["warnings"].append(f"Training file lacks ID column: {args.id_col}")
    if args.id_col not in pred.columns:
        report["warnings"].append(f"Prediction file lacks ID column: {args.id_col}")
    if len(shared_loci) == 0:
        report["problems"].append("No shared locus columns detected between train and predict files")
    elif len(shared_loci) < 100:
        report["warnings"].append(f"Only {len(shared_loci)} locus columns detected. This looks like 7-locus MLST, not cgMLST/wgMLST.")

    counts = report["train_summary_after_exclusions"]["source_counts"] or {}
    low = {k: v for k, v in counts.items() if v < args.min_source_size}
    if low:
        report["warnings"].append(f"Some training classes are below min_source_size={args.min_source_size}: {low}")

    if train_delim != "tab" or pred_delim != "tab":
        report["warnings"].append("SourceRunnerML reads sep='\\t'; use the normalised TSV files for the actual run.")

    if report["problems"]:
        report["status"] = "FAIL"

    report_json = outdir / "source_runner_preflight_report.json"
    report_txt = outdir / "source_runner_preflight_report.txt"
    with report_json.open("w") as f:
        json.dump(report, f, indent=2)

    with report_txt.open("w") as f:
        f.write(f"SourceRunnerML preflight status: {report['status']}\n\n")
        f.write(f"Train: {train_path} ({train.shape[0]} rows, delimiter={train_delim})\n")
        f.write(f"Predict: {pred_path} ({pred.shape[0]} rows, delimiter={pred_delim})\n")
        f.write(f"Training rows after exclusions: {train_clean.shape[0]}\n")
        f.write(f"Source column: {args.source_col}\n")
        f.write(f"Detected/shared loci ({len(shared_loci)}): {', '.join(shared_loci)}\n\n")
        f.write("Training source counts after exclusions:\n")
        for k, v in counts.items():
            f.write(f"  {k}\t{v}\n")
        f.write("\nMissingness by locus, train:\n")
        for k, v in report["train_summary_after_exclusions"]["missingness_by_locus"].items():
            f.write(f"  {k}\t{v:.4%}\n")
        f.write("\nMissingness by locus, predict:\n")
        for k, v in report["predict_summary"]["missingness_by_locus"].items():
            f.write(f"  {k}\t{v:.4%}\n")
        if report["warnings"]:
            f.write("\nWarnings:\n")
            for w in report["warnings"]:
                f.write(f"  - {w}\n")
        if report["problems"]:
            f.write("\nProblems:\n")
            for p in report["problems"]:
                f.write(f"  - {p}\n")

    if args.write_tsv:
        train_tsv = outdir / f"{train_path.stem}.sourcerunner.train.tsv"
        pred_tsv = outdir / f"{pred_path.stem}.sourcerunner.predict.tsv"
        write_tsv(train_clean, train_tsv)
        write_tsv(pred, pred_tsv)
        print(f"Wrote: {train_tsv}")
        print(f"Wrote: {pred_tsv}")

    print(f"Preflight status: {report['status']}")
    print(f"Report: {report_txt}")
    print(f"Shared loci: {len(shared_loci)}")
    if report["warnings"]:
        print("Warnings:")
        for w in report["warnings"]:
            print(f"  - {w}")
    if report["problems"]:
        print("Problems:")
        for p in report["problems"]:
            print(f"  - {p}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
