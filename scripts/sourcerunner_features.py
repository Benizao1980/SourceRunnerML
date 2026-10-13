"""Shared genomic locus selection and fit-on-training-partition preprocessing.

No biological distance is implied by the numeric magnitude of allele IDs.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer


def read_locus_file(path):
    """Read a one-locus-per-line text/TSV list; optional 'locus' header and comments."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    names = [line.split("\t", 1)[0].strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if names and names[0].lower() in ("locus", "loci", "locus_id"):
        names = names[1:]
    if not names:
        raise ValueError("Locus file must contain at least one locus name")
    if len(set(names)) != len(names):
        raise ValueError("Locus file contains duplicate locus names")
    return names


def select_shared_loci(train, predict, *, loci_file="", loci_pattern="", loci_prefix="", loci=""):
    """Select common loci, with precedence file > regex > prefix > legacy list.

    Explicit file lists are strict: every selected locus must exist in both
    tables. Regex and prefix modes return the common intersection.
    """
    for name, frame in (("training", train), ("prediction", predict)):
        if frame.columns.duplicated().any():
            raise ValueError(f"Duplicate column name in {name} table; rename or drop duplicates")
    if loci_file:
        method = "file"
        selected = read_locus_file(loci_file)
        absent_train = sorted(set(selected) - set(train.columns))
        absent_pred = sorted(set(selected) - set(predict.columns))
        if absent_train or absent_pred:
            raise ValueError(
                f"Locus-file entries absent from input tables: train={absent_train[:20]}, "
                f"predict={absent_pred[:20]}"
            )
    elif loci_pattern:
        method = "regex"
        rgx = re.compile(loci_pattern)
        selected = [c for c in train.columns if rgx.match(c) and c in predict.columns]
    elif loci_prefix:
        method = "prefix"
        selected = [c for c in train.columns if c.startswith(loci_prefix) and c in predict.columns]
    else:
        method = "legacy_list"
        names = [x.strip() for x in loci.split(",") if x.strip()]
        selected = [x for x in names if x in train.columns and x in predict.columns]
    if not selected:
        raise ValueError("No shared genomic loci selected; check --loci_file/--loci_pattern/--loci_prefix and column names")
    return selected, method


class AllelePreprocessor(BaseEstimator, TransformerMixin):
    """Filter missing loci and impute using ONLY the partition passed to fit()."""

    def __init__(self, missingness=0.20):
        self.missingness = missingness

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("AllelePreprocessor requires a pandas DataFrame with locus column names")
        if not 0 <= self.missingness <= 1:
            raise ValueError("missingness must be in [0,1]")
        if X.empty:
            raise ValueError("Cannot fit on an empty training partition")
        self.retained_loci_ = [
            c for c in X.columns
            if X[c].isna().mean() <= self.missingness and X[c].notna().any()
        ]
        if not self.retained_loci_:
            raise ValueError("No loci retained after training-partition missingness filtering")
        self.imputer_ = SimpleImputer(strategy="most_frequent")
        self.imputer_.fit(X.loc[:, self.retained_loci_])
        return self

    def transform(self, X):
        if not hasattr(self, "imputer_"):
            raise ValueError("AllelePreprocessor has not been fitted")
        absent = set(self.retained_loci_) - set(X.columns)
        if absent:
            raise ValueError(f"Input missing fitted loci: {sorted(absent)[:20]}")
        return np.asarray(self.imputer_.transform(X.loc[:, self.retained_loci_]), dtype=float)

    def get_feature_names_out(self, input_features=None):
        if not hasattr(self, "retained_loci_"):
            raise ValueError("AllelePreprocessor has not been fitted")
        return np.asarray(self.retained_loci_, dtype=object)
