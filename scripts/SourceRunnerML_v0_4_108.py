#!/usr/bin/env python3

import os
import time
import logging
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
# Import global variables from utils:
from utils_v0_4_108 import (
    setup_logging,
    save_pickle,
    load_config,
    parse_loci,
    filter_missingness,
    get_classifier,
    auto_select_classifier,
    tune_classifier,
    bayesian_tune_classifier,
    stacking_ensemble_refined,
    run_bootstrap,
    average_report,
    summarize_non_loci_columns,
    generate_roc_curves,
    generate_precision_recall_curve,
    generate_calibration_plot,
    generate_multiclass_calibration_plots,
    predict_with_bootstrap,
    add_per_class_summaries,
    summarize_metadata,
    plot_prediction_probabilities,
    compute_locus_metrics,
    resample_data,
    plot_stacked_heatmap,
    plot_stacked_bar,
    phylogeny_impute,
    one_in_one_out_cv,
    generate_allelic_matrix,
    compartmentalized_predictions,
    smart_phylogeny_impute,
    generate_itol_files,
    generate_microreact_files,
    plot_structure_bar,
    plot_pca,
    evaluate_external_test_set,
    save_checkpoint,
    refined_uncertainty,
    plot_uncertainty_histogram,
    HAS_SHAP,
    HAS_PLOTLY
)

# Additional imports required for ensemble models:
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")
warnings.filterwarnings("ignore", message=".*Auto-choosing.*")
warnings.filterwarnings("ignore", message=".*TBB Warning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="SourceRunnerML v0_4_108")
    parser.add_argument('--train_file', required=True)
    parser.add_argument('--predict_file')
    parser.add_argument('--loci_prefix', default='CAMP')
    parser.add_argument('--loci_pattern', default="")  # Optional regex for locus detection
    parser.add_argument('--missing_values', choices=['drop', 'impute'], default='impute')
    parser.add_argument('--missingness', type=float, default=0.2)
    parser.add_argument('--classifier', choices=['random_forest', 'xgboost', 'logreg', 'lightgbm', 'catboost', 'ensemble'], default='random_forest')
    parser.add_argument('--auto_classifier', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=0)
    parser.add_argument('--pred_bootstrap', type=int, default=None, help="Bootstrap replicates for predictions; defaults to training bootstrap")
    parser.add_argument('--burn_in', type=int, default=0)
    parser.add_argument('--cpus', type=int, default=8)
    parser.add_argument('--resample_method', choices=['none', 'undersample', 'oversample', 'smote'], default='none')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--min_confidence', type=float, default=0.5)
    parser.add_argument('--ensemble_method', choices=['none', 'voting', 'stacking'], default='none')
    parser.add_argument('--bayesian_tune', action='store_true', help="Enable Bayesian hyperparameter tuning")
    parser.add_argument('--smart_impute', action='store_true', help="Enable smart phylogeny-informed imputation using MLST data")
    parser.add_argument('--output_locus_metrics', action='store_true', help="Output detailed locus metrics (Fst, Tajima's D, etc.)")
    parser.add_argument('--mode', choices=['quick-run', 'diagnostic', 'comprehensive'], default='diagnostic')
    parser.add_argument('--export_itol', action='store_true', help="Export iTOL-compatible files")
    parser.add_argument('--export_microreact', action='store_true', help="Export Microreact-compatible files")
    parser.add_argument('--newick_tree', default=None, help="File path for Newick tree to be used for iTOL export")
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning (GridSearchCV placeholder)")
    parser.add_argument('--metadata_summary', action='store_true', help="Generate metadata summary output")
    parser.add_argument('--save_checkpoint', action='store_true', help="Save intermediate results for resuming")
    parser.add_argument('--config', default=None, help="Path to JSON configuration file")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--meta_learner', default=None, help="Optional meta-learner for stacking ensemble (e.g., 'logreg')")
    parser.add_argument('--external_test_file', default=None, help="Path to external test set file for validation")
    return parser.parse_args()

def apply_mode_settings(args):
    if args.mode == 'quick-run':
        args.bootstrap = 5
        args.resample_method = "none"
    elif args.mode == 'diagnostic':
        args.bootstrap = 20
    elif args.mode == 'comprehensive':
        args.bootstrap = 50
        args.resample_method = "smote"
    logging.info(f"Mode set to: {args.mode}")
    return args

def main():
    try:
        start = time.time()
        args = parse_args()
        if args.config:
            import json
            with open(args.config, 'r') as cf:
                config = json.load(cf)
            for key, value in config.items():
                setattr(args, key, value)
            logging.info("Configuration file applied.")
        args = apply_mode_settings(args)
        if args.pred_bootstrap is None:
            args.pred_bootstrap = args.bootstrap
        args.output_dir = f"outputs/{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logging(args)
        
        logging.info("Input Arguments:")
        for arg, value in vars(args).items():
            logging.info(f"  {arg}: {value}")
        
        # Read the training data only once.
        train_df = pd.read_csv(args.train_file, sep='\t', dtype=str, low_memory=False)
        train_df = train_df.reset_index(drop=True)
        if args.loci_pattern:
            args.loci_prefix = ""
        loci = parse_loci(args, train_df)
        loci = filter_missingness(train_df, loci, args.missingness)
        logging.info(f"Proceeding with {len(loci)} loci after missingness filtering")
        
        # Convert allele columns to numeric in a unified step
        train_df[loci] = train_df[loci].apply(pd.to_numeric, errors='coerce')
        if args.missing_values == 'impute':
            train_df[loci] = pd.DataFrame(
                SimpleImputer(strategy='most_frequent').fit_transform(train_df[loci]),
                columns=loci
            )
        else:
            train_df = train_df.dropna(subset=loci).reset_index(drop=True)
        
        # Derive both X and y from the same DataFrame
        X = pd.DataFrame(train_df[loci], columns=loci)
        source_col = [c for c in train_df.columns if c.lower() == 'source'][0]
        y = train_df[source_col]
        # Ensure indices align
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        assert X.shape[0] == y.shape[0], "Mismatch between X and y row counts after preprocessing."
        
        if args.smart_impute:
            X = smart_phylogeny_impute(X, train_df)
        elif args.mode in ['diagnostic', 'comprehensive']:
            X = phylogeny_impute(X)
        
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        logging.info("Training matrix loaded and labels encoded successfully.")
        
        if args.resample_method != "none":
            X, y_encoded = resample_data(X, y_encoded, method=args.resample_method)
            logging.info(f"Resampled training data using method: {args.resample_method}")
        
        if args.output_locus_metrics:
            locus_metrics = compute_locus_metrics(X, y)
            locus_metrics.to_csv(f"{args.output_dir}/{args.run_name}_locus_metrics.tsv", sep="\t", index=False)
            logging.info("Locus metrics saved.")
        allelic_matrix = generate_allelic_matrix(X)
        allelic_matrix.to_csv(f"{args.output_dir}/{args.run_name}_allelic_matrix.tsv", sep="\t", index=False)
        
        if args.auto_classifier:
            clf_name, auto_score = auto_select_classifier(X, y_encoded)
            logging.info(f"Auto-selected classifier: {clf_name} with accuracy {auto_score:.4f}")
        else:
            clf_name = args.classifier
        
        if args.ensemble_method == 'stacking':
            meta_learner = None
            if args.meta_learner and args.meta_learner.lower() == 'logreg':
                meta_learner = LogisticRegression(max_iter=1000, solver='lbfgs')
            final_model = stacking_ensemble_refined(X, y_encoded, meta_learner)
            clf_name = "stacking_" + clf_name
        elif args.ensemble_method == 'voting':
            final_model = VotingClassifier(estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)),
                ('logreg', LogisticRegression(max_iter=1000, solver='lbfgs')),
                ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', verbosity=0)),
                ('lgb', lgb.LGBMClassifier(random_state=42, verbose=-1)),
                ('cat', CatBoostClassifier(verbose=0, random_state=42))
            ], voting='soft')
            clf_name = "voting_" + clf_name
        else:
            final_model = get_classifier(clf_name)
        
        if args.bayesian_tune:
            from sklearn.ensemble import StackingClassifier
            if isinstance(final_model, StackingClassifier):
                logging.warning("Bayesian tuning is not supported for stacking ensembles. Skipping tuning.")
            else:
                logging.info("Performing Bayesian hyperparameter tuning...")
                param_space = {'n_estimators': (50, 200)}
                final_model = bayesian_tune_classifier(final_model, param_space, X, y_encoded)
                clf_name = clf_name + "_bayes"
        elif args.tune:
            logging.info("Tuning hyperparameters using GridSearchCV...")
            base_clf = final_model
            param_grid = {'n_estimators': [50, 100, 200]}
            final_model = tune_classifier(base_clf, param_grid, X, y_encoded)
            clf_name = clf_name + "_tuned"
        else:
            final_model.fit(X, y_encoded)
        
        if args.bootstrap:
            logging.info(f"Running {args.bootstrap} bootstraps (burn-in: {args.burn_in}) on {args.cpus} CPU(s)")
            reports, cms, best_model = run_bootstrap(X, y_encoded, loci, clf_name, args)
            report_summary, report_std = average_report(reports, burn_in=args.burn_in)
            overall_accuracy = report_summary.get("accuracy", None)
            overall_accuracy_std = report_std.get("accuracy", None)
            f1_scores = {le.classes_[i]: report_summary[i]["f1-score"] for i in range(len(le.classes_)) if i in report_summary}
            f1_scores_std = {le.classes_[i]: report_std[i]["f1-score"] for i in range(len(le.classes_)) if i in report_std}
            avg_cm = np.mean(cms, axis=0)
        else:
            report_summary = classification_report(y_encoded, final_model.predict(X), output_dict=True)
            overall_accuracy = report_summary.get("accuracy", None)
            f1_scores = {le.classes_[i]: report_summary[str(i)]["f1-score"] for i in range(len(le.classes_)) if str(i) in report_summary}
            avg_cm = confusion_matrix(y_encoded, final_model.predict(X))
        
        # Refined uncertainty quantification
        uncertainty_metrics = refined_uncertainty(final_model, X, pd.Series(y_encoded))
        logging.info(f"Refined uncertainty metrics: {uncertainty_metrics}")
        
        if args.retrain:
            logging.info("Retraining final model on full training data.")
            final_model = get_classifier(clf_name)
            final_model.fit(X, y_encoded)
            save_pickle(final_model, f"{args.output_dir}/{args.run_name}_retrained_clf.p")
        
        final_preds_numeric = final_model.predict(X)
        final_preds = le.inverse_transform(final_preds_numeric)
        final_cm = confusion_matrix(y_encoded, final_preds_numeric)
        clf_report = classification_report(y_encoded, final_preds_numeric, output_dict=True)
        
        with open(f"{args.output_dir}/{args.run_name}_training_report.txt", "w") as f:
            f.write("SourceRunnerML v0_4_108\n")
            f.write("Input Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"  {arg}: {value}\n")
            f.write(f"\nTraining samples: {X.shape[0]}\n")
            f.write(f"Loci used: {X.shape[1]}\n")
            f.write(f"Classes: {list(le.classes_)}\n")
            if overall_accuracy is not None:
                f.write(f"\nSelf-test Accuracy (Bootstrap Average): {overall_accuracy:.4f}")
                if overall_accuracy_std is not None:
                    f.write(f" ± {overall_accuracy_std:.4f}\n")
                else:
                    f.write("\n")
            if f1_scores:
                f.write("\nF1 Scores per Class:\n")
                for cls in le.classes_:
                    if cls in f1_scores:
                        std_val = f1_scores_std.get(cls, 0)
                        f.write(f"  {cls}: {f1_scores[cls]:.4f} ± {std_val:.4f}\n")
            f.write("\nConfusion Matrix (Final Model on Training Data - Real vs Predicted):\n")
            f.write("Rows: Actual, Columns: Predicted\n")
            f.write(np.array2string(final_cm, separator=', '))
            f.write("\n\nDetailed Classification Report:\n")
            f.write(pd.DataFrame(clf_report).to_string())
        
        if args.metadata_summary:
            metadata_summary = summarize_metadata(train_df)
            metadata_summary.to_csv(f"{args.output_dir}/{args.run_name}_metadata_summary.tsv", sep="\t")
        else:
            logging.info("Metadata summary not generated (flag --metadata_summary not set).")
        
        if HAS_SHAP:
            try:
                import shap
                shap_values = shap.Explainer(final_model, X)
                shap_vals = shap_values(X)
                import matplotlib.pyplot as plt
                shap.summary_plot(shap_vals, X, show=False)
                shap_plot_path = os.path.join(args.output_dir, "shap_summary.png")
                plt.savefig(shap_plot_path, dpi=300)
                plt.close()
                logging.info("SHAP summary plot saved.")
            except Exception as e:
                logging.warning(f"SHAP computation failed: {e}")
        
        if args.external_test_file:
            ext_df = pd.read_csv(args.external_test_file, sep='\t', dtype=str, low_memory=False)
            ext_df = ext_df.reset_index(drop=True)
            ext_loci = [col for col in ext_df.columns if col.startswith(args.loci_prefix)]
            ext_df[ext_loci] = ext_df[ext_loci].apply(pd.to_numeric, errors='coerce')
            if args.missing_values == 'impute':
                ext_df[ext_loci] = pd.DataFrame(
                    SimpleImputer(strategy='most_frequent').fit_transform(ext_df[ext_loci]),
                    columns=ext_loci
                )
            else:
                ext_df = ext_df.dropna(subset=ext_loci).reset_index(drop=True)
            X_ext = pd.DataFrame(ext_df[ext_loci], columns=ext_loci)
            source_col_ext = [c for c in ext_df.columns if c.lower() == 'source'][0]
            y_ext = ext_df[source_col_ext]
            y_ext_encoded = le.transform(y_ext)
            evaluate_external_test_set(final_model, X_ext, y_ext_encoded, args.output_dir)
        
        if args.predict_file:
            logging.info("Starting prediction block...")
            pred_df = pd.read_csv(args.predict_file, sep='\t', dtype=str, low_memory=False)
            pred_df = pred_df.reset_index(drop=True)
            metadata_cols = [col for col in pred_df.columns if not col.startswith('CAMP')]
            pred_df[loci] = pred_df[loci].apply(pd.to_numeric, errors='coerce')
            if args.missing_values == 'impute':
                pred_df[loci] = pd.DataFrame(
                    SimpleImputer(strategy='most_frequent').fit_transform(pred_df[loci]),
                    columns=loci
                )
            else:
                pred_df = pred_df.dropna(subset=loci).reset_index(drop=True)
            probs, preds_numeric, ci_low, ci_high = predict_with_bootstrap(pd.DataFrame(pred_df[loci], columns=loci), final_model, args.pred_bootstrap, args.cpus)
            preds = le.inverse_transform(preds_numeric)
            metadata_df = pred_df[metadata_cols].copy()
            metadata_df['Predicted_Source'] = preds
            metadata_df['Confidence'] = probs.max(axis=1)
            predicted_class_indices = np.argmax(probs, axis=1)
            ci_lower_final = [float(ci_low[i, idx]) for i, idx in enumerate(predicted_class_indices)]
            ci_upper_final = [float(ci_high[i, idx]) for i, idx in enumerate(predicted_class_indices)]
            metadata_df['CI_Lower'] = ci_lower_final
            metadata_df['CI_Upper'] = ci_upper_final
            for i, cls in enumerate(le.classes_):
                metadata_df[f"Prob_{cls}"] = probs[:, i]
            metadata_df.loc[metadata_df['Confidence'] < args.min_confidence, 'Predicted_Source'] = 'Uncertain'
            pred_out_path = f"{args.output_dir}/{args.run_name}_srml_out.tsv"
            logging.info(f"Writing predictions to {pred_out_path}")
            metadata_df.to_csv(pred_out_path, sep="\t", index=False)
            
            prediction_counts = metadata_df['Predicted_Source'].value_counts()
            total_preds = prediction_counts.sum()
            prediction_percentages = (prediction_counts / total_preds * 100).round(2)
            prob_columns = [f"Prob_{cls}" for cls in le.classes_]
            prob_summary = metadata_df[prob_columns].describe()
            with open(f"{args.output_dir}/{args.run_name}_prediction_summary.txt", "w") as f:
                f.write("Input Arguments:\n")
                for arg, value in vars(args).items():
                    f.write(f"  {arg}: {value}\n")
                f.write("\nPrediction Counts (Raw and %):\n")
                for label in prediction_counts.index:
                    f.write(f"  {label}: {prediction_counts[label]} ({prediction_percentages[label]}%)\n")
                f.write("\n\nPrediction Probabilities Summary:\n")
                f.write(prob_summary.to_string())
                if np.allclose(ci_low, ci_high):
                    f.write("\n\nNote: Upper and lower confidence intervals are identical, indicating minimal bootstrap variance.")
            
            metadata_columns_to_breakdown = ['country', 'continent', 'year', 'ST', 'clonal_complex']
            for col in metadata_columns_to_breakdown:
                if col in metadata_df.columns:
                    breakdown = metadata_df.groupby(col)['Predicted_Source'].value_counts(normalize=True) * 100
                    breakdown = breakdown.round(2)
                    breakdown.to_csv(f"{args.output_dir}/{args.run_name}_prediction_{col}_breakdown.tsv", sep="\t")
            
            plot_prediction_probabilities(probs, args.output_dir, le.classes_)
            plot_stacked_heatmap(probs, args.output_dir, le.classes_)
            plot_stacked_bar(probs, args.output_dir, le.classes_)
            plot_structure_bar(probs, args.output_dir, le.classes_, run_name=args.run_name)
            plot_pca(X, train_df[source_col].values, args.output_dir, run_name=args.run_name, prefix="allele_pca", title_suffix="(Actual Source)")
            pred_probs_df = pd.DataFrame(probs, columns=le.classes_)
            plot_pca(pred_probs_df, preds, args.output_dir, run_name=args.run_name, prefix="pred_pca", title_suffix="(Predicted Source)")
            
            mean_preds, std_preds = compartmentalized_predictions(final_model, pd.DataFrame(pred_df[loci], columns=loci), n_folds=5, cpus=args.cpus)
            comp_summary = pd.DataFrame({
                "Mean_Probability": np.mean(mean_preds, axis=1),
                "Std_Probability": np.mean(std_preds, axis=1)
            })
            comp_summary.to_csv(f"{args.output_dir}/{args.run_name}_compartmentalized_prediction_summary.tsv", sep="\t")
            
            plot_uncertainty_histogram(std_preds, args.output_dir, run_name=args.run_name)
            
            if args.export_itol and args.newick_tree is not None:
                with open(args.newick_tree, 'r') as nt:
                    newick_tree = nt.read().strip()
                generate_itol_files(newick_tree, metadata_df, args.output_dir)
            
            if args.export_microreact:
                generate_microreact_files(metadata_df, args.output_dir)
            
            if HAS_PLOTLY:
                try:
                    y_pred_proba_df = pd.DataFrame(probs, columns=le.classes_)
                    if 'source' in pred_df.columns:
                        y_true_df = pd.get_dummies(pred_df['source'])
                        generate_roc_curves(y_true_df, y_pred_proba_df, args.output_dir)
                    else:
                        logging.info("True labels not available in prediction file for interactive ROC generation.")
                except Exception as e:
                    logging.warning(f"Interactive ROC plot generation failed: {e}")
        
        if args.save_checkpoint:
            save_checkpoint(final_model, args.output_dir)
        
        logging.info(f"✅ Run completed in {time.time() - start:.2f} sec")
    
    except Exception as e:
        logging.exception("Unhandled exception occurred:")
        raise

if __name__ == "__main__":
    main()
