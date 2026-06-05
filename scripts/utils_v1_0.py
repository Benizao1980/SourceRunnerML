import os
import re
from pathlib import Path
from typing import Optional, Union
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, LeaveOneOut
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             auc, precision_recall_curve, accuracy_score)
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
from scipy.stats import entropy

# For additional resampling
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# For interactive visualizations with Plotly (optional)
try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# For explainability using SHAP (optional)
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# ----------------------------
# Custom Color Palette and Colormap
# ----------------------------
CUSTOM_PALETTE = {
    "yellow": "#FFEF55",
    "green": "#2E8B57",
    "purple": "#B000FF",
    "orange": "#F24F26",
    "pink": "#FF69B4",
    "blue": "#0097C3",
    "skyblue": "#9BE3F9",
    "coral": "#FF7F50",   
    "peach": "#FFDAB9"    
}
import matplotlib.colors as mcolors
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "custom_palette",
    [CUSTOM_PALETTE["yellow"], CUSTOM_PALETTE["green"], CUSTOM_PALETTE["orange"], CUSTOM_PALETTE["blue"], CUSTOM_PALETTE["pink"]],
    N=256
)

# ----------------------------
# Basic Utility Functions
# ----------------------------
def setup_logging(args=None):
    logging.basicConfig(level=logging.INFO)
    logging.info("Logging is set up.")

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    logging.info(f"Object successfully saved to {file_path}")

# ----------------------------
# Configuration File Support
# ----------------------------
def load_config(config_path):
    import json
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info(f"Configuration loaded from {config_path}")
    return config

# ----------------------------
# Source Label Processing
# ----------------------------
def process_source_labels(df, source_col='source', min_source_size=None, merge_map=None, logger=None):
    """
    Normalises, merges, and filters source labels for training.
    Returns:
        filtered_df, retained_labels, excluded_labels
    """
    import pandas as pd
    df = df.copy()
    if merge_map is None:
        merge_map = {
            "cow": "ruminants",
            "cattle": "ruminants",
            "sheep": "ruminants",
            "goat": "ruminants",
            "beef": "ruminants",
            "broiler chicken": "chicken",
            "hen": "chicken",
            "swine": "pig",
            "hog": "pig",
            "boar": "pig",
            "pork": "pig",
            "duck": "wild bird",
            "quail": "poultry",
            "goose": "wild bird",
            "turkey": "poultry"
        }  # add more synonyms if needed  
    
    # Apply merge mapping
    if merge_map:
        df[source_col] = df[source_col].str.strip().str.lower().map(merge_map).fillna(df[source_col])

    counts = df[source_col].value_counts()
    retained = counts[counts >= min_source_size].index.tolist()
    excluded = counts[counts < min_source_size]

    if logger:
        logger.info("Source label distribution before filtering:")
        logging.info("Unique raw source labels: %s", df[source_col].unique().tolist())
        for label, count in counts.items():
            logger.info(f"  {label}: {count}")

        logger.info("Retaining the following source categories:")
        for label in retained:
            logger.info(f"  {label}: {counts[label]} isolates")

        if not excluded.empty:
            logger.warning("Excluding source categories with insufficient counts:")
            for label, count in excluded.items():
                logger.warning(f"  {label}: {count}")

    # Filter the dataframe
    df = df[df[source_col].isin(retained)].copy()
    return df, retained, excluded

# ----------------------------
# Locus Processing
# ----------------------------
def _auto_detect_prefix(df, min_cols=7):
    """
    Returns the most frequent prefix (before the first non-alpha character)
    that appears in >= min_cols columns, else raises ValueError.
    """
    from collections import Counter
    import re
    prefixes = Counter()
    for col in df.columns:
        m = re.match(r"([A-Za-z]+)", col)
        if m:
            prefixes[m.group(1)] += 1
    if not prefixes:
        raise ValueError("Could not detect any alphabetic prefixes.")
    best, count = prefixes.most_common(1)[0]
    if count < min_cols:
        raise ValueError(f"No prefix occurs in ≥{min_cols} columns (best was {best} with {count}).")
    return best

def parse_loci(args, df):
    """
    Returns the list of locus columns based on user prefix / regex / auto-detect.
    """
    import re
    if args.loci_pattern:
        loci = [c for c in df.columns if re.match(args.loci_pattern, c)]
    else:
        prefix = args.loci_prefix or _auto_detect_prefix(df)
        loci = [c for c in df.columns if c.startswith(prefix)]
    if not loci:
        raise ValueError("No locus columns detected – check --loci_prefix / --loci_pattern")
    return loci

def filter_missingness(df, loci, threshold):
    missing = df[loci].isna().mean()
    kept = [locus for locus in loci if missing[locus] <= threshold]
    return kept

# ----------------------------
# Classifier Functions and Tuning
# ----------------------------
def get_classifier(name, n_jobs=1, class_weights=None):
    name = name.lower().strip()
    if name.startswith(("stacking_", "voting_")):
        name = name.split("_", 1)[1]
    if name == "xgbclassifier":
        name = "xgboost"

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            n_jobs=n_jobs,
            random_state=17,
            class_weight="balanced",
            oob_score=True,
        )

    if name == "logreg":
        return LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
        )

    if name == "xgboost":
        return XGBClassifier(            
            use_label_encoder=False,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=n_jobs,
        )

    if name == "lightgbm":
        return lgb.LGBMClassifier(
            random_state=42,
            verbose=-1,
            n_estimators=300,
            class_weight="balanced",
            n_jobs=n_jobs,
        )

    if name == "catboost":
        return CatBoostClassifier(
            verbose=0,
            random_state=42,
            class_weights=class_weights,
        )

    if name == "ensemble":
        estimators = [
            ("rf", get_classifier("random_forest", n_jobs)),
            ("logreg", get_classifier("logreg", n_jobs)),
            ("xgb", get_classifier("xgboost", n_jobs)),  
            ("lgb", get_classifier("lightgbm", n_jobs)),
            ("cat", get_classifier("catboost", n_jobs)),
        ]
        return VotingClassifier(estimators=estimators, voting="soft")

    raise ValueError(f"Unsupported classifier: {name}")



from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import logging

def _safe_stratified_cv(y, requested=3, random_state=42):
    # Encode labels to count per-class samples robustly
    le = LabelEncoder()
    y_enc = le.fit_transform(np.asarray(y).ravel())
    # Smallest class size bounds the number of splits
    min_class = int(np.min(np.bincount(y_enc)))
    # Also guard against tiny datasets
    n_splits = max(2, min(requested, len(y_enc), min_class))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), n_splits

def auto_select_classifier(X, y):
    """
    Try several classifiers and pick the best by CV accuracy.
    Robust to tiny/imbalanced classes and individual model failures.
    """
    classifiers = ['random_forest', 'logreg', 'xgboost', 'lightgbm', 'catboost']
    scores = {}
    errors = {}

    # Defensive cleaning: replace infs left over after coercion/imputation
    try:
        X_ = X.replace([np.inf, -np.inf], np.nan)
        if X_.isna().any().any():
            X_ = X_.fillna(0)
    except Exception:
        # If X is a numpy array
        X_ = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cv, eff_splits = _safe_stratified_cv(y, requested=3)
    logging.info(f"[Auto] Using StratifiedKFold with n_splits={eff_splits}")

    for clf_name in classifiers:
        clf = get_classifier(clf_name)
        try:
            fold_scores = cross_val_score(
                clf, X_, y, cv=cv, scoring='accuracy', n_jobs=-1, error_score=np.nan
            )
            mean_score = np.nanmean(fold_scores)
            if np.isnan(mean_score):
                raise ValueError(f"All CV folds returned NaN for {clf_name}")
            logging.info(f"[Auto] {clf_name} accuracy: {mean_score:.4f}")
            scores[clf_name] = mean_score
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            logging.warning(f"[Auto] Skipping {clf_name} due to error -> {msg}")
            errors[clf_name] = msg

    if not scores:
        # Nothing worked: raise a clear, actionable error
        details = "; ".join([f"{k}: {v}" for k, v in errors.items()]) or "no models attempted"
        raise ValueError(f"Auto-selection failed: all models errored. Details: {details}")

    best = max(scores, key=scores.get)
    return best, scores[best]

    classifiers = ['random_forest', 'logreg', 'xgboost', 'lightgbm', 'catboost']
    scores = {}
    for clf_name in classifiers:
        clf = get_classifier(clf_name)
        score = cross_val_score(clf, X, y, cv=3).mean()
        logging.info(f"[Auto] {clf_name} accuracy: {score:.4f}")
        scores[clf_name] = score
    best = max(scores, key=scores.get)
    return best, scores[best]

def tune_classifier(clf, param_grid, X, y, cv=3):
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X, y)
    logging.info(f"Best parameters: {grid.best_params_}")
    logging.info(f"Best score: {grid.best_score_:.4f}")
    return grid.best_estimator_

def bayesian_tune_classifier(clf, param_space, X, y, n_iter=30, cv=3):
    from skopt import BayesSearchCV
    bayes_cv = BayesSearchCV(clf, param_space, cv=cv, n_iter=n_iter, scoring='f1_weighted', n_jobs=-1, random_state=42)
    bayes_cv.fit(X, y)
    logging.info(f"Bayesian tuning best parameters: {bayes_cv.best_params_}")
    logging.info(f"Bayesian tuning best score: {bayes_cv.best_score_:.4f}")
    return bayes_cv.best_estimator_

def stacking_ensemble_refined(X, y, meta_learner=None, dynamic_weighting=False):
    if meta_learner is None:
        meta_learner = LogisticRegression(max_iter=1000, solver='lbfgs')
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)),
        ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs')),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)),
        ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1)),
        ('cat', CatBoostClassifier(verbose=0, random_state=42))
    ]
    stacked_clf = StackingClassifier(estimators=estimators, final_estimator=meta_learner, cv=5, passthrough=True)
    logging.info("Refined stacking ensemble created with meta-learner: " + str(meta_learner))
    return stacked_clf

# ----------------------------
# Advanced Data Imputation
# ----------------------------
def smart_phylogeny_impute(X, metadata):
    logging.info("Performing advanced smart phylogeny-informed imputation using 'ST' and 'clonal_complex' if available.")
    if 'ST' not in metadata.columns:
        logging.info("No 'ST' column found; skipping smart imputation.")
        return X
    X_imputed = X.copy()
    overall_modes = X.mode().iloc[0]
    group_cols = ['ST']
    if 'clonal_complex' in metadata.columns:
        group_cols.append('clonal_complex')
    group_modes = metadata.groupby(group_cols).agg(lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan)
    for col in X.columns:
        def impute_value(row):
            if pd.isna(row[col]):
                key = tuple(metadata.loc[row.name, gc] for gc in group_cols)
                if len(group_cols) == 1:
                    key = key[0]
                if key in group_modes.index and not pd.isna(group_modes.loc[key, col]):
                    return group_modes.loc[key, col]
                else:
                    return overall_modes[col]
            else:
                return row[col]
        X_imputed[col] = X_imputed.apply(impute_value, axis=1)
    logging.info("Advanced smart imputation completed.")
    return X_imputed

def phylogeny_impute(X):
    logging.info("Basic phylogeny-informed imputation not yet implemented. Returning original data.")
    return X

# ----------------------------
# Refined Uncertainty Quantification
# ----------------------------
def refined_uncertainty(model, X, y):
    if hasattr(model, 'oob_score_'):
        oob_error = 1 - model.oob_score_
        logging.info(f"OOB Error: {oob_error:.4f}")
        return oob_error
    else:
        preds, accuracy, std_probs = one_in_one_out_cv(model, X, y)
        logging.info(f"LOOCV Accuracy: {accuracy:.4f}")
        return accuracy, std_probs

def plot_uncertainty_histogram(std_preds, output_dir, run_name="SRML"):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    overall_std = np.mean(std_preds, axis=1)
    plt.figure(figsize=(8,6))
    # Using "coral" from our updated palette
    plt.hist(overall_std, bins=20, color=CUSTOM_PALETTE["coral"], edgecolor='black')
    plt.xlabel("Mean Std Dev of Predicted Probabilities", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.title(f"Uncertainty Histogram ({run_name})", fontsize=16)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{run_name}_uncertainty_histogram.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Uncertainty histogram saved to {out_path}")

# ----------------------------
# Bootstrapping & Self-Testing
# ----------------------------
def run_bootstrap(X, y, loci, clf_name, args):
    from joblib import Parallel, delayed, parallel_backend
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import time, numpy as np

    logging.info(f"🧠 Starting bootstrap with {args.bootstrap} replicates "
                 f"using {args.cpus} CPU(s)…")
    logging.info("Shape of X before bootstrapping: %s", X.shape)
    start_time = time.time()

    skf = StratifiedKFold(n_splits=args.bootstrap,
                          shuffle=True, random_state=42)

    # ----- helper for one replicate ------------------------------------------
    def get_target(vec, idx):
        try:    return vec.iloc[idx]
        except AttributeError:
                 return vec[idx]

    def fit_and_report(train_idx, test_idx):
        clf = get_classifier(clf_name)
        # pin each model to one thread
        if hasattr(clf, "n_jobs"):
            clf.set_params(n_jobs=1)
        if hasattr(clf, "thread_count"):      # CatBoost
            clf.set_params(thread_count=1)

        clf.fit(X.iloc[train_idx], get_target(y, train_idx))
        preds = clf.predict(X.iloc[test_idx])
        rep = classification_report(get_target(y, test_idx),
                                    preds, output_dict=True)
        cm  = confusion_matrix(get_target(y, test_idx), preds)
        return rep, cm, clf
    # ------------------------------------------------------------------------

    with parallel_backend("threading", n_jobs=args.cpus):
        results = []
        splits = list(skf.split(X, y))
        for i in range(0, len(splits), args.cpus):      # chunked execution
            batch = splits[i : i + args.cpus]
            results.extend(
                Parallel()(
                    delayed(fit_and_report)(tr, te) for tr, te in batch
                )
            )

    reports     = [r  for r, cm, clf in results]
    cms         = [cm for r, cm, clf in results]
    best_model  = results[0][2]

    logging.info("🧪 Bootstrapping completed in %.2f sec",
                 time.time() - start_time)
    return reports, cms, best_model

def average_report(reports, burn_in=0):
    valid = reports[burn_in:]
    avg_report = {}
    std_report = {}
    keys = valid[0].keys()
    for key in keys:
        if isinstance(valid[0][key], dict):
            avg_report[key] = {}
            std_report[key] = {}
            for metric in valid[0][key]:
                values = [r[key][metric] for r in valid]
                avg_report[key][metric] = np.mean(values)
                std_report[key][metric] = np.std(values)
        else:
            values = [r[key] for r in valid]
            avg_report[key] = np.mean(values)
            std_report[key] = np.std(values)
    return avg_report, std_report

def one_in_one_out_cv(model, X, y):
    from sklearn.model_selection import LeaveOneOut
    loo = LeaveOneOut()
    predictions = []
    pred_probs = []
    supports_proba = hasattr(model, "predict_proba")
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        clf = get_classifier(model.__class__.__name__.lower())
        clf.fit(X_train, y_train)
        predictions.append(clf.predict(X_test)[0])
        if supports_proba:
            pred_probs.append(clf.predict_proba(X_test)[0])
    predictions = np.array(predictions)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, predictions)
    if supports_proba:
        pred_probs = np.array(pred_probs)
        std_probs = np.std(pred_probs, axis=0)
    else:
        std_probs = None
    logging.info(f"Leave-One-Out CV Accuracy: {accuracy:.4f}")
    return predictions, accuracy, std_probs

# ----------------------------
# Reporting & Visualizations
# ----------------------------
def summarize_non_loci_columns(df):
    non_loci_columns = [col for col in df.columns if not col.startswith('CAMP')]
    summary_stats = df[non_loci_columns].describe(include='all').T
    return summary_stats

# utils_v0_4_116.py
# ---------------------------------------------------------------------------
def compute_locus_metrics(X: pd.DataFrame,
                          y: pd.Series,
                          out_path: Union[Path, str, None] = None) -> pd.DataFrame:
    """
    Per-locus information metrics.

    Parameters
    ----------
    X : DataFrame   allele matrix (rows = samples, cols = loci)
    y : Series      source labels (length must match X)
    out_path: Union[Path, str, None]
        If given, write the resulting table as TSV.

    Returns
    -------
    DataFrame with one row per locus and columns:
        - missing_prop
        - n_unique
        - mutual_information
        - class_entropy_mean
        - f_score
        - Fst (placeholder, NaN)
        - Tajima_D (placeholder, NaN)
    """
    if len(X) != len(y):
        raise ValueError("X rows and y length differ "
                         f"({len(X)} vs {len(y)}).")

    metrics = pd.DataFrame(index=X.columns)
    metrics["missing_prop"] = X.isna().mean()
    metrics["n_unique"]     = X.nunique(dropna=True)

    # ------------------------------------------------------------------
    # Replace NaN allele calls with sentinel and make everything numeric
    X_clean = X.fillna(-1)

    # helper to encode any dtype as integer codes
    def encode(series):
        return pd.factorize(series.astype(str), sort=False)[0]

    y_enc = encode(y)
    
    # --- Mutual information -------------------------------------------------
    mi = [mutual_info_score(y_enc, encode(X_clean[c])) for c in X.columns]
    metrics["mutual_information"] = mi

    # --- Class-conditional entropy -----------------------------------------
    def per_class_entropy(col):
        return (pd.concat([col, y], axis=1)
                  .groupby(y.name)[col.name]
                  .value_counts(normalize=True)
                  .groupby(level=0)
                  .apply(lambda v: entropy(v)))
    ent = X.apply(per_class_entropy).T.mean(axis=1)
    metrics["class_entropy_mean"] = ent

    # --- ANOVA F-score (categorical → codes) --------------------------
    X_enc = np.vstack([encode(X_clean[c]) for c in X.columns]).T
    f_val, _ = f_classif(X_enc, y_enc)
    metrics["f_score"] = f_val

    # --- Placeholders -------------------------------------------------------
    metrics["Fst"]       = np.nan   # TODO: real Fst
    metrics["Tajima_D"]  = np.nan   # TODO: real Tajima's D

    if out_path:
        metrics.to_csv(out_path, sep="\t")
        logging.info("Locus metrics written → %s", out_path)

    return metrics

def generate_allelic_matrix(X):
    logging.info("Generating allelic matrix (placeholder: returning input unchanged).")
    return X.copy()

def compartmentalized_predictions(model, X, n_folds=5, cpus=1):
    from joblib import Parallel, delayed
    preds = []
    for i in range(n_folds):
        preds.append(model.predict_proba(X))
    preds = np.array(preds)
    mean_preds = np.mean(preds, axis=0)
    std_preds = np.std(preds, axis=0)
    logging.info("Compartmentalized predictions computed (placeholder).")
    return mean_preds, std_preds

def generate_itol_files(newick_tree, metadata_df, output_dir):
    out_path = os.path.join(output_dir, "itol_tree.txt")
    with open(out_path, "w") as f:
         f.write(newick_tree)
    logging.info(f"iTOL file generated at {out_path}")

def generate_microreact_files(metadata_df, output_dir):
    out_path = os.path.join(output_dir, "microreact_metadata.tsv")
    metadata_df.to_csv(out_path, sep="\t", index=False)
    logging.info(f"Microreact file generated at {out_path}")

def add_per_class_summaries(predictions, y_true):
    df = pd.DataFrame({'True': y_true, 'Predicted': predictions})
    summary = df.groupby(['True', 'Predicted']).size().unstack(fill_value=0)
    logging.info("Per-class summary generated.")
    return summary

def summarize_metadata(df):
    non_loci_columns = [col for col in df.columns if not col.startswith('CAMP')]
    summary = df[non_loci_columns].describe(include='all').T
    logging.info("Metadata summary computed.")
    return summary

def generate_roc_curves(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    palette = list(CUSTOM_PALETTE.values())
    for i, class_name in enumerate(y_true.columns):
        fpr, tpr, _ = roc_curve(y_true[class_name], y_pred_proba[class_name])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color=palette[i % len(palette)], lw=2, label=f'ROC (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color=CUSTOM_PALETTE["peach"], lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Curve for {class_name}', fontsize=16)
        plt.legend(loc="lower right", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{class_name}_roc_curve.png"), dpi=300)
        plt.close()
        logging.info(f"ROC curve for {class_name} saved.")

def generate_precision_recall_curve(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    palette = list(CUSTOM_PALETTE.values())
    for i, class_name in enumerate(y_true.columns):
        precision, recall, _ = precision_recall_curve(y_true[class_name], y_pred_proba[class_name])
        plt.figure()
        plt.plot(recall, precision, color=palette[i % len(palette)], lw=2, label='PR Curve')
        plt.xlabel('Recall', fontsize=14)
        plt.ylabel('Precision', fontsize=14)
        plt.title(f'Precision-Recall Curve for {class_name}', fontsize=16)
        plt.legend(loc="lower left", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{class_name}_pr_curve.png"), dpi=300)
        plt.close()
        logging.info(f"Precision-Recall curve for {class_name} saved.")

def generate_calibration_plot(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure()
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Mean predicted probability', fontsize=14)
    plt.ylabel('Fraction of positives', fontsize=14)
    plt.title('Calibration Plot', fontsize=16)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration_plot.png"), dpi=300)
    plt.close()
    logging.info("Calibration plot saved.")

def generate_multiclass_calibration_plots(y_true, y_pred_proba, classes, output_dir):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    for i, cls in enumerate(classes):
        y_true_bin = (y_true == cls).astype(int)
        prob_pred = y_pred_proba[:, i]
        prob_true, prob_pred_curve = calibration_curve(y_true_bin, prob_pred, n_bins=10)
        plt.figure()
        plt.plot(prob_pred_curve, prob_true, marker='o', linewidth=1, label='Calibration curve')
        plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
        plt.xlabel('Mean predicted probability', fontsize=14)
        plt.ylabel('Fraction of positives', fontsize=14)
        plt.title(f'Calibration Plot for {cls}', fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{cls}_calibration_plot.png"), dpi=300)
        plt.close()
        logging.info(f"Calibration plot for {cls} saved.")

def plot_prediction_probabilities(probs, output_dir, classes):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    n_classes = len(classes)
    fig, axs = plt.subplots(n_classes, 1, figsize=(8, 3 * n_classes))
    palette = list(CUSTOM_PALETTE.values())
    if n_classes == 1:
        axs = [axs]
    for i, cls in enumerate(classes):
        axs[i].hist(probs[:, i], bins=20, color=palette[i % len(palette)], edgecolor='black')
        axs[i].set_title(f"Predicted Probabilities for {cls}", fontsize=14)
        axs[i].set_xlabel("Probability", fontsize=12)
        axs[i].set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_prediction_histograms.png"), dpi=300)
    plt.close()
    logging.info("Combined prediction histograms saved.")

def plot_stacked_heatmap(pred_probs, output_dir, classes):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_theme(style="whitegrid", context="talk")
    except ImportError:
        logging.warning("Seaborn not installed, skipping stacked heatmap.")
        return
    df_heat = pd.DataFrame(pred_probs, columns=classes)
    plt.figure(figsize=(max(8, 0.1 * df_heat.shape[0]), 4))
    ax = sns.heatmap(df_heat.T, cmap=CUSTOM_CMAP, cbar_kws={'label': 'Predicted Probability (%)'})
    plt.xlabel("Isolate Index", fontsize=14)
    plt.ylabel("Source", fontsize=14)
    plt.title("Per-Isolate Predicted Source Distribution", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_heatmap_predicted_sources.png"), dpi=300)
    plt.close()
    logging.info("Stacked heatmap saved.")

def plot_stacked_bar(pred_probs, output_dir, classes):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    df = pd.DataFrame(pred_probs, columns=classes)
    df_percentage = df.div(df.sum(axis=1), axis=0) * 100
    plt.figure(figsize=(max(10, 0.05 * df_percentage.shape[0]), 6))
    df_percentage.plot(kind='bar', stacked=True, colormap=CUSTOM_CMAP)
    plt.xlabel("Isolate Index", fontsize=14)
    plt.ylabel("Percentage Attribution", fontsize=14)
    plt.title("100% Stacked Bar Chart of Predicted Source Probabilities", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_bar_chart_predictions.png"), dpi=300)
    plt.close()
    logging.info("Stacked bar chart saved.")

def plot_structure_bar(pred_probs, output_dir, classes, run_name="SRML"):
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    df_probs = pd.DataFrame(pred_probs, columns=classes)
    df_percentage = df_probs.div(df_probs.sum(axis=1), axis=0) * 100
    n_isolates = df_percentage.shape[0]
    fig, ax = plt.subplots(figsize=(max(10, 0.08 * n_isolates), 6))
    bottom = np.zeros(n_isolates)
    colors = [CUSTOM_PALETTE[key] for key in CUSTOM_PALETTE]
    for i, cls in enumerate(classes):
        ax.bar(range(n_isolates),
               df_percentage[cls],
               bottom=bottom,
               color=colors[i % len(colors)],
               width=1.0)
        bottom += df_percentage[cls].values
    ax.set_xlim([0, n_isolates])
    ax.set_ylim([0, 100])
    ax.set_xlabel("Isolate Index", fontsize=14)
    ax.set_ylabel("Predicted Source (%)", fontsize=14)
    ax.set_title(f"Structure-Style Plot for {run_name}", fontsize=16)
    ax.legend(classes, bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{run_name}_structure_bar.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"Structure-style bar plot saved to {out_path}")

def plot_pca(data, labels, output_dir, run_name="SRML", prefix="pca", title_suffix=""):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import seaborn as sns
    sns.set_theme(style="whitegrid", context="talk")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(data)
    unique_labels = np.unique(labels)
    colors = [CUSTOM_PALETTE[key] for key in CUSTOM_PALETTE]
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(unique_labels):
        idx = (labels == label)
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=colors[i % len(colors)],
                   label=label,
                   edgecolor='k',
                   alpha=0.7,
                   s=40)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)", fontsize=14)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)", fontsize=14)
    ax.set_title(f"PCA Plot {title_suffix} ({run_name})", fontsize=16)
    ax.legend(fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"{prefix}_plot.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    logging.info(f"PCA plot saved to {out_path}")

def evaluate_external_test_set(model, X_test, y_test, output_dir):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds)
    report_path = os.path.join(output_dir, "external_test_report.txt")
    with open(report_path, "w") as f:
        f.write("External Test Set Evaluation Report\n")
        f.write(pd.DataFrame(report).to_string())
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(cm, separator=', '))
    logging.info("External test set evaluation saved.")
    return report, cm

def save_checkpoint(model, output_dir, checkpoint_name="final_model_checkpoint.p"):
    save_pickle(model, os.path.join(output_dir, checkpoint_name))

def resample_data(X, y, method='none'):
    """
    Resample the data to address class imbalance.
    """
    if method == 'none':
        return X, y
    elif method == 'undersample':
        from imblearn.under_sampling import RandomUnderSampler
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        return X_res, y_res
    elif method == 'oversample':
        from imblearn.over_sampling import RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X, y)
        return X_res, y_res
    elif method == 'smote':
        if not HAS_SMOTE:
            raise ImportError("SMOTE is not installed. Please install imbalanced-learn.")
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    else:
        raise ValueError("Unsupported resampling method")
    
def predict_with_bootstrap(X, model, bootstrap, cpus):
    from joblib import Parallel, delayed
    X_numeric = X.astype(float)
    if hasattr(model, "predict_proba"):
        all_probs = Parallel(n_jobs=cpus)(delayed(model.predict_proba)(X_numeric) for _ in range(bootstrap))
        all_probs = np.array(all_probs, dtype=float)
        avg_probs = np.mean(all_probs, axis=0)
        std_probs = np.std(all_probs, axis=0)
        preds = np.array([model.classes_[np.argmax(avg_probs[i])] for i in range(avg_probs.shape[0])])
        ci_low = avg_probs - std_probs
        ci_high = avg_probs + std_probs
        return avg_probs, preds, ci_low, ci_high
    else:
        all_preds = Parallel(n_jobs=cpus)(delayed(model.predict)(X_numeric) for _ in range(bootstrap))
        all_preds = np.array(all_preds)
        preds = all_preds[0]
        return None, preds, None, None
