import os
import re
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
    "coral": "#FF7F50",   # Added missing key for coral
    "peach": "#FFDAB9"    # Added missing key for peach
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
    # Normalize the classifier name to handle different casings or variants
    name = name.lower().strip()
    if name == 'xgbclassifier':
        name = 'xgboost'
    # Remove any stacking or voting prefixes to get the base classifier name
    if name.startswith("stacking_"):
        name = name[len("stacking_"):]
    if name.startswith("voting_"):
        name = name[len("voting_"):]
    
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    elif name == 'logreg':
        return LogisticRegression(max_iter=1000, solver='lbfgs')
    elif name == 'xgboost':
        return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)
    elif name == 'lightgbm':
        return lgb.LGBMClassifier(random_state=42, verbose=-1)
    elif name == 'catboost':
        return CatBoostClassifier(verbose=0, random_state=42)
    elif name == 'ensemble':
        # Example: Create a voting ensemble from a set of classifiers.
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)),
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
    from joblib import Parallel, delayed
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    import time

    logging.info(f"🧠 Starting bootstrap with {args.bootstrap} replicates using {args.cpus} CPU(s)...")
    logging.info("Shape of X before bootstrapping: " + str(X.shape))
    start_time = time.time()
    skf = StratifiedKFold(n_splits=args.bootstrap, shuffle=True, random_state=42)
    
    def get_target(y, indices):
        try:
            return y.iloc[indices]
        except AttributeError:
            return y[indices]
    
    def fit_and_report(train_idx, test_idx):
        clf = get_classifier(clf_name)
        clf.fit(X.iloc[train_idx], get_target(y, train_idx))
        preds = clf.predict(X.iloc[test_idx])
        rep = classification_report(get_target(y, test_idx), preds, output_dict=True)
        cm = confusion_matrix(get_target(y, test_idx), preds)
        return rep, cm, clf
    
    results = Parallel(n_jobs=args.cpus)(
        delayed(fit_and_report)(train_idx, test_idx) for train_idx, test_idx in skf.split(X, y)
    )
    reports = [r for r, cm, clf in results]
    cms = [cm for r, cm, clf in results]
    best_model = results[0][2]
    elapsed = time.time() - start_time
    logging.info(f"🧪 Bootstrapping completed in {elapsed:.2f} sec")
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

def compute_locus_metrics(X, y):
    """
    Compute per-locus metrics: Fst (placeholder), Tajima's D (placeholder), and Mutual Information.
    If the allele vector length doesn't match y, skip the locus.
    """
    mi_scores = []
    loci_list = []
    for locus in X.columns:
        if len(X[locus]) != len(y):
            logging.warning(f"Skipping {locus} due to length mismatch: {len(X[locus])} vs {len(y)}")
            continue
        try:
            mi = mutual_info_score(y.values, X[locus].values)
            mi_scores.append(mi)
            loci_list.append(locus)
        except Exception as e:
            logging.warning(f"Error computing MI for {locus}: {e}")
    metrics = pd.DataFrame({
         "locus": loci_list,
         "Fst": np.nan,
         "Tajima_D": np.nan,
         "Mutual_Information": mi_scores
    })
    logging.info("Locus metrics computed with mutual information scores.")
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

# Only one version of summarize_metadata is retained:
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
