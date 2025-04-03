import os
import sys
import re
import logging
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix, roc_curve, 
                             auc, precision_recall_curve)
from sklearn.metrics import mutual_info_score

# For additional resampling (if available)
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ----------------------------
# Basic Utility Functions
# ----------------------------
def setup_logging(args=None):
    logging.basicConfig(level=logging.INFO)
    logging.info("Logging is set up.")

def save_pickle(obj, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object successfully saved to {file_path}")

# ----------------------------
# Locus Processing
# ----------------------------
def parse_loci(args, df):
    if hasattr(args, "loci_pattern") and args.loci_pattern:
        pattern = re.compile(args.loci_pattern)
        loci = [col for col in df.columns if pattern.search(col)]
        logging.info(f"Auto-detected {len(loci)} loci using regex pattern '{args.loci_pattern}'")
    else:
        loci = [col for col in df.columns if col.startswith(args.loci_prefix)]
        logging.info(f"Auto-detected {len(loci)} loci using prefix '{args.loci_prefix}'")
    return loci

def filter_missingness(df, loci, threshold):
    missing = df[loci].isna().mean()
    kept = [locus for locus in loci if missing[locus] <= threshold]
    return kept

# ----------------------------
# Classifier Functions and Tuning
# ----------------------------
def get_classifier(name):
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == 'logreg':
        return LogisticRegression(max_iter=1000, solver='lbfgs')
    elif name == 'xgboost':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    elif name == 'lightgbm':
        return lgb.LGBMClassifier(random_state=42, verbose=-1)
    elif name == 'catboost':
        return CatBoostClassifier(verbose=0, random_state=42)
    elif name == 'ensemble':
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs')),
            ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)),
            ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1)),
            ('cat', CatBoostClassifier(verbose=0, random_state=42))
        ]
        return VotingClassifier(estimators=estimators, voting='soft')
    else:
        raise ValueError(f"Unsupported classifier: {name}")

def auto_select_classifier(X, y):
    from sklearn.model_selection import cross_val_score
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
    # Advanced tuning: currently using GridSearchCV; Bayesian optimization can be added later.
    grid = GridSearchCV(clf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    grid.fit(X, y)
    logging.info(f"Best parameters: {grid.best_params_}")
    logging.info(f"Best score: {grid.best_score_:.4f}")
    return grid.best_estimator_

# ----------------------------
# Bootstrapping & Self-Testing
# ----------------------------
def run_bootstrap(X, y, loci, clf_name, args):
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import time
    logging.info(f"🧠 Starting bootstrap with {args.bootstrap} replicates using {args.cpus} CPU(s)...")
    start_time = time.time()
    skf = StratifiedKFold(n_splits=args.bootstrap, shuffle=True, random_state=42)
    reports = []
    cms = []
    def fit_and_report(train_idx, test_idx):
        clf = get_classifier(clf_name)
        clf.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = clf.predict(X.iloc[test_idx])
        rep = classification_report(y.iloc[test_idx], preds, output_dict=True)
        cm = confusion_matrix(y.iloc[test_idx], preds)
        return rep, cm, clf
    results = Parallel(n_jobs=args.cpus)(
        delayed(fit_and_report)(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)
    )
    reports = [r for r, cm, clf in results]
    cms = [cm for r, cm, clf in results]
    best_model = results[0][2]
    logging.info(f"🧪 Bootstrapping completed in {time.time() - start_time:.2f} sec")
    return reports, cms, best_model

def average_report(reports, burn_in=0):
    valid = reports[burn_in:]
    avg_report = {}
    keys = valid[0].keys()
    for key in keys:
        if isinstance(valid[0][key], dict):
            avg_report[key] = {}
            for metric in valid[0][key]:
                avg_report[key][metric] = np.mean([r[key][metric] for r in valid])
        else:
            avg_report[key] = np.mean([r[key] for r in valid])
    return avg_report

# ----------------------------
# Reporting & Visualizations
# ----------------------------
def summarize_non_loci_columns(df):
    non_loci_columns = [col for col in df.columns if not col.startswith('CAMP')]
    summary_stats = df[non_loci_columns].describe(include='all').T
    return summary_stats

def generate_roc_curves(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    for class_name in y_true.columns:
        fpr, tpr, _ = roc_curve(y_true[class_name], y_pred_proba[class_name])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {class_name}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, f"{class_name}_roc_curve.png"))
        plt.close()

def generate_precision_recall_curve(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    for class_name in y_true.columns:
        precision, recall, _ = precision_recall_curve(y_true[class_name], y_pred_proba[class_name])
        plt.figure()
        plt.plot(recall, precision, lw=2, label='PR Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve for {class_name}')
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(output_dir, f"{class_name}_pr_curve.png"))
        plt.close()

def generate_calibration_plot(y_true, y_pred_proba, output_dir):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    plt.figure()
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Calibration curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
    plt.xlabel('Mean predicted probability')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Plot')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "calibration_plot.png"))
    plt.close()

def plot_prediction_probabilities(probs, output_dir, classes):
    import matplotlib.pyplot as plt
    n_classes = len(classes)
    fig, axs = plt.subplots(n_classes, 1, figsize=(8, 3 * n_classes))
    if n_classes == 1:
        axs = [axs]
    for i, cls in enumerate(classes):
        axs[i].hist(probs[:, i], bins=20, color='skyblue', edgecolor='black')
        axs[i].set_title(f"Predicted Probabilities for {cls}")
        axs[i].set_xlabel("Probability")
        axs[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_prediction_histograms.png"))
    plt.close()

def plot_stacked_heatmap(pred_probs, output_dir, classes):
    import matplotlib.pyplot as plt
    import seaborn as sns
    df_heat = pd.DataFrame(pred_probs, columns=classes)
    plt.figure(figsize=(max(8, 0.1 * df_heat.shape[0]), 4))
    ax = sns.heatmap(df_heat.T, cmap="YlGnBu", cbar_kws={'label': 'Predicted Probability (%)'})
    plt.xlabel("Isolate Index")
    plt.ylabel("Source")
    plt.title("Per-Isolate Predicted Source Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "stacked_heatmap_predicted_sources.png"))
    plt.close()

def process_and_output_predictions(predictions, df, output_dir, report, classifier_name):
    logging.info("process_and_output_predictions is deprecated in v4_4_0.")
    pass

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

def add_per_class_summaries(report, classifier_name):
    return report

def summarize_metadata(df):
    metadata_cols = [col for col in df.columns if not col.startswith('CAMP')]
    summary = df[metadata_cols].describe(include='all').T
    return summary

def compute_locus_metrics(X, y):
    metrics = []
    for col in X.columns:
        mi = mutual_info_score(y, X[col])
        fst = np.nan  # Placeholder for Fst calculation; to implement allelic matrix conversion
        metrics.append({"locus": col, "mutual_info": mi, "fst": fst})
    return pd.DataFrame(metrics)

def resample_data(X, y, method="none"):
    from sklearn.utils import resample
    df = X.copy()
    df['source'] = y
    classes = df['source'].unique()
    if method == "undersample":
        min_count = df['source'].value_counts().min()
        resampled = []
        for cls in classes:
            subset = df[df['source'] == cls]
            subset_downsampled = resample(subset, replace=False, n_samples=min_count, random_state=42)
            resampled.append(subset_downsampled)
        df_resampled = pd.concat(resampled)
    elif method == "oversample":
        max_count = df['source'].value_counts().max()
        resampled = []
        for cls in classes:
            subset = df[df['source'] == cls]
            subset_upsampled = resample(subset, replace=True, n_samples=max_count, random_state=42)
            resampled.append(subset_upsampled)
        df_resampled = pd.concat(resampled)
    elif method == "smote" and HAS_SMOTE:
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        return X_res, y_res
    else:
        return X, y
    X_resampled = df_resampled.drop(columns=['source'])
    y_resampled = df_resampled['source']
    return X_resampled, y_resampled

def phylogeny_impute(X):
    logging.info("Advanced phylogeny-informed imputation not yet implemented. Returning original data.")
    return X

def one_in_one_out_cv(X, y):
    logging.info("One-in, one-out CV not yet implemented. Using standard cross-validation.")
    return None

def generate_allelic_matrix(X):
    logging.info("Generating allelic matrix for Fst estimation (placeholder).")
    return X.copy()

def compartmentalized_predictions(model, X, n_folds=5, cpus=1):
    from sklearn.model_selection import KFold
    from joblib import Parallel, delayed
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    predictions = []
    for train_index, test_index in kf.split(X):
        X_fold = X.iloc[test_index]
        if hasattr(model, "predict_proba"):
            preds = model.predict_proba(X_fold)
        else:
            preds = model.predict(X_fold)
        predictions.append(preds)
    predictions = np.array(predictions)
    mean_preds = np.mean(predictions, axis=0)
    std_preds = np.std(predictions, axis=0)
    return mean_preds, std_preds
