#!/usr/bin/env python3
__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import argparse, logging, time, warnings, os
from pathlib import Path

import numpy as np, pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


from utils_v1_0 import (
    setup_logging, parse_loci, filter_missingness, process_source_labels,
    resample_data, phylogeny_impute, smart_phylogeny_impute,
    auto_select_classifier, get_classifier,
    run_bootstrap, average_report,
    predict_with_bootstrap, evaluate_external_test_set,
    compute_locus_metrics, summarize_metadata, save_pickle,
    stacking_ensemble_refined, refined_uncertainty, generate_allelic_matrix,
    plot_prediction_probabilities, plot_stacked_heatmap, plot_stacked_bar,
    plot_structure_bar, plot_pca, HAS_SHAP, HAS_PLOTLY,
)

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"SourceRunnerML {__version__}")

    # I/O
    p.add_argument('--train_file', required=True)
    p.add_argument('--predict_file')
    p.add_argument('--external_test_file')
    p.add_argument('--run_name', required=True)
    p.add_argument('--output_dir', default='outputs')

    # Loci / missingness
    p.add_argument('--loci_prefix')
    p.add_argument('--loci_pattern', default='')
    p.add_argument('--missingness', type=float, default=0.20)
    p.add_argument('--missing_values', choices=['drop','impute'], default='impute')
    p.add_argument('--smart_impute', action='store_true')

    # Labels / resampling
    p.add_argument('--source_col', default='source')
    p.add_argument('--min_source_size', type=int, default=100)
    p.add_argument('--merge_map')
    p.add_argument('--resample_method',
                   choices=['none','oversample','undersample','smote'],
                   default='none',
                   help='Class balancing at data level (rarely needed when '
                    'class_weight=\"balanced\" is used).')

    # Modelling
    p.add_argument('--classifier', default='random_forest',
                   choices=['random_forest','xgboost','logreg',
                            'lightgbm','catboost','ensemble'])
    p.add_argument('--auto_classifier', action='store_true')
    p.add_argument('--ensemble_method', choices=['none','voting','stacking'],
                   default='none')
    p.add_argument('--bootstrap', type=int, default=0)
    p.add_argument('--burn_in', type=int, default=0)
    p.add_argument('--retrain', action='store_true')
    p.add_argument('--pred_bootstrap', type=int, default=1)

    # Prediction filtering
    p.add_argument('--min_confidence', type=float, default=0.0)

    # Optional / not-yet-implemented flags (accepted but ignored)
    p.add_argument('--tune', action='store_true')
    p.add_argument('--bayesian_tune', action='store_true')
    p.add_argument('--meta_learner')
    p.add_argument('--export_itol', action='store_true')
    p.add_argument('--newick_tree')
    p.add_argument('--export_microreact', action='store_true')

    # Reporting & misc
    p.add_argument('--output_locus_metrics', action='store_true')
    p.add_argument('--metadata_summary', action='store_true')
    p.add_argument('--mode',
                   choices=['quick-run','diagnostic','comprehensive'],
                   default='diagnostic')
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('--debug', action='store_true')
    p.add_argument('--skip_selftest', action='store_true', help='Skip internal CV self-test to save time on large runs')
    p.add_argument('--make_structure_plot', action='store_true', help='Generate STRUCTURE-like stacked bar plot for predictions')
    return p.parse_args()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_out_dir(base: str, name: str) -> Path:
    d = Path(base) / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    d.mkdir(parents=True, exist_ok=True)
    return d

def warn_unused(args):
    for flag in ('tune','bayesian_tune','meta_learner',
                 'export_itol','export_microreact','newick_tree'):
        if getattr(args, flag):
            logging.warning("Flag --%s is accepted but not yet implemented.",
                            flag.replace('_','-'))

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start = time.time()
    args = parse_args()
    out_dir = make_out_dir(args.output_dir, args.run_name)
    setup_logging(args)
    warn_unused(args)
    logging.info("🗃  Outputs → %s", out_dir)

    # 1. LOAD
    df = pd.read_csv(args.train_file, sep='\t', dtype=str, low_memory=False)

    # 2. LOCI
    if args.loci_pattern:
        args.loci_prefix = ''
    loci = filter_missingness(df, parse_loci(args, df), args.missingness)
    logging.info("Using %d loci (≤%.1f%% missing)", len(loci), args.missingness*100)
    X = df[loci].apply(pd.to_numeric, errors='coerce')

    # 3. LABELS
    # If the user did NOT supply --merge_map we pass an *empty* dict,
    # which tells utils to keep source labels exactly as listed.
    supplied_map = None
    if args.merge_map and Path(args.merge_map).exists():
        supplied_map = dict(pd.read_csv(args.merge_map, sep='\t', header=None).values)
    else:
        supplied_map = {}          # <- disables synonym merging
    df, _, _ = process_source_labels(df, args.source_col,
                                    args.min_source_size,
                                    merge_map=supplied_map,
                                    logger=logging)

    X = X.loc[df.index]
    y_series = df[args.source_col]
    le = LabelEncoder().fit(y_series)
    y = le.transform(y_series)

    # 4. MISSING-VALUE HANDLING
    if args.missing_values == 'impute':
        imp = SimpleImputer(strategy='most_frequent')
        X[:] = imp.fit_transform(X)
    else:
        keep = X.dropna().index
        X, y = X.loc[keep], y[keep]

    # --- optional phylogeny-aware imputation ---------------------------------
    if args.smart_impute:
        logging.info("Performing smart phylogeny-informed imputation …")
        X = smart_phylogeny_impute(X, df)          # needs metadata
    elif args.mode in ('diagnostic', 'comprehensive'):
        X = phylogeny_impute(X)                    # plain version, ONE argument

    # 5. RESAMPLING
    if args.resample_method != 'none':
        logging.warning(
            "You selected --resample_method %s, but class_weight=\"balanced\" is "
            "already active for all classifiers. Data-level oversampling may be "
            "unnecessary and memory-intensive.", args.resample_method)
        X, y = resample_data(X, y, method=args.resample_method)
        logging.info("Resampled via %s", args.resample_method)

    # 6. CLASSIFIER
    # Balanced weights just for CatBoost
    from sklearn.utils import class_weight
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y),
        y=y
    ).tolist()

    # Pick the model name
    clf_name = (auto_select_classifier(X, y)[0]
                if args.auto_classifier else args.classifier)

    # --- stacking ensemble ----------------------------------------------------
    if args.ensemble_method == "stacking":
        model = stacking_ensemble_refined(X, y)
        clf_name = "stack_" + clf_name

    # --- voting ensemble ------------------------------------------------------
    elif args.ensemble_method == "voting":
        from sklearn.ensemble import VotingClassifier
        def _make(clf):
            return get_classifier(
                clf,
                n_jobs=args.cpus,
                class_weights=(class_weights if clf == "catboost" else None),
            )

        model = VotingClassifier(
            estimators=[(n, _make(n))
                        for n in ["random_forest", "logreg",
                                  "xgboost", "lightgbm", "catboost"]],
            voting="soft",
            n_jobs=args.cpus,
        )
        clf_name = "vote_" + clf_name

    # --- default classifier ---------------------------------------------------
    else:
        model = get_classifier(
            clf_name,
            n_jobs=args.cpus,
            class_weights=(class_weights if clf_name == "catboost" else None),
        )

    # 7. BOOTSTRAP / TRAIN
    if args.bootstrap:
        reports, cms, model = run_bootstrap(X, y, loci, clf_name, args)
        avg_acc = average_report(reports, args.burn_in)[0].get('accuracy', np.nan)
        logging.info("Bootstrap mean acc %.3f", avg_acc)
        # Save full bootstrap reports and confusion matrices
        import json
        boot_json = out_dir / f"{args.run_name}_training_bootstrap_reports.json"
        with open(boot_json, "w") as f:
            json.dump(reports, f, indent=2)
        cms_json = out_dir / f"{args.run_name}_training_confusion_matrices.json"
        try:
            # cms returned as a list of arrays; convert to lists
            cms_lists = [cm.tolist() for cm in cms]
        except Exception:
            cms_lists = []
        with open(cms_json, "w") as f:
            json.dump(cms_lists, f, indent=2)
        # Save averaged training summary
        avg_report, std_report = average_report(reports, args.burn_in)
        with open(out_dir / f"{args.run_name}_training_summary.json", "w") as f:
            json.dump({"avg": avg_report, "std": std_report}, f, indent=2)

        if args.retrain:  model.fit(X, y)
    else:
        model.fit(X, y)
    save_pickle(model, out_dir / f"{args.run_name}_model.pkl")
    # 7b. SELF-TEST (Stratified CV on training set)
    if not args.skip_selftest:
        try:
            # y is already label-encoded above; do not transform it again.
            y_enc = np.asarray(y, dtype=int)
            min_class = int(np.min(np.bincount(y_enc)))
            n_splits = max(2, min(5, len(y_enc), min_class))
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

            y_pred = cross_val_predict(get_classifier(clf_name, n_jobs=args.cpus), X, y_enc, cv=skf, n_jobs=1)
            acc = accuracy_score(y_enc, y_pred)
            numeric_labels = list(range(len(le.classes_)))
            cm = confusion_matrix(y_enc, y_pred, labels=numeric_labels)
            rep = classification_report(y_enc, y_pred, labels=numeric_labels, target_names=list(le.classes_), output_dict=True, zero_division=0)

            import json
            with open(out_dir / f"{args.run_name}_selftest_report.json", "w") as f:
                json.dump({"accuracy": acc, "report": rep, "labels": list(le.classes_)}, f, indent=2)
            np.savetxt(out_dir / f"{args.run_name}_selftest_confusion_matrix.tsv", cm, fmt="%d", delimiter="\t")
            logging.info(f"Self-test CV: acc={acc:.3f}; saved report + confusion matrix.")
        except Exception as e:
            logging.warning(f"Self-test evaluation failed: {e}")
    else:
        logging.info("Skipping self-test CV (--skip_selftest).")

    # 7c. BASIC TRAINING REPORT
    report_path = out_dir / f"{args.run_name}_training_report.txt"
    try:
        with report_path.open("w") as fh:
            fh.write(f"SourceRunnerML {__version__}\n\n")
            fh.write("Input Arguments:\n")
            for k, v in vars(args).items():
                fh.write(f"  {k}: {v}\n")
            fh.write(f"\nTraining rows retained: {len(y)}\n")
            fh.write(f"Loci retained: {len(loci)}\n")
            fh.write(f"Classes: {list(le.classes_)}\n")
        logging.info("Training report written → %s", report_path)
    except Exception as e:
        logging.warning("Could not write training report: %s", e)

    # 8. METRICS
    if args.output_locus_metrics:
        compute_locus_metrics(
            df[loci],
            df[args.source_col],
            out_dir / f"{args.run_name}_locus_metrics.tsv"
        )
    
    if args.metadata_summary:
        summary = summarize_metadata(df)                       # returns a DataFrame
        summary.to_csv(out_dir / f"{args.run_name}_metadata.txt",
                        sep="\t", index=False)
        logging.info("Metadata summary written → %s",
                    out_dir / f"{args.run_name}_metadata.txt")
    
        # 9. PREDICTION
    if args.predict_file:
        p_df     = pd.read_csv(args.predict_file, sep='\t', dtype=str, low_memory=False)
        p_X      = p_df[loci].apply(pd.to_numeric, errors='coerce')
        if args.missing_values == 'impute':  p_X[:] = imp.transform(p_X)
        probs, pred_num, *_ = predict_with_bootstrap(
            p_X, model, bootstrap=max(1,args.pred_bootstrap), cpus=args.cpus)
        cls      = le.inverse_transform(pred_num)
        max_prob = probs.max(axis=1)
        filt_cls = np.where(max_prob < args.min_confidence, 'Uncertain', cls)
        out      = p_df.copy()
        out['predicted_source']  = cls
        out['max_probability']   = max_prob
        out['predicted_filtered'] = filt_cls
        out.to_csv(out_dir / f"{args.run_name}_predictions.tsv",
                   sep='\t', index=False)
        logging.info("Predictions written")

        # Save full per-class probabilities (detailed prediction summary)
        probs_df = pd.DataFrame(probs, columns=le.classes_)
        probs_df.insert(0, 'sample_id', p_df.index.astype(str))
        probs_df.to_csv(out_dir / f"{args.run_name}_prediction_probabilities.tsv",
                        sep='\t', index=False)
        logging.info("Per-class prediction probabilities written.")
        # STRUCTURE-like stacked bar plot; disabled by default for large prediction sets.
        if args.make_structure_plot:
            try:
                plot_structure_bar(probs, out_dir, list(le.classes_), run_name=args.run_name)
                logging.info("STRUCTURE-like stacked bar plot saved.")
            except Exception as e:
                logging.warning("Could not generate STRUCTURE-like plot: %s", e)

    # 10. EXTERNAL TEST
    if args.external_test_file:
        t_df = pd.read_csv(args.external_test_file, sep='\t', dtype=str, low_memory=False)
        t_X  = t_df[loci].apply(pd.to_numeric, errors='coerce')
        if args.missing_values == 'impute':  t_X[:] = imp.transform(t_X)
        if args.source_col in t_df.columns:
            y_test = le.transform(t_df[args.source_col])
            evaluate_external_test_set(model, t_X, y_test, out_dir)
        else:
            logging.warning("External test lacks labels; skipping evaluation")

    logging.info("✅  Finished in %.1f s", time.time() - start)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
