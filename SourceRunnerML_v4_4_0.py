#!/usr/bin/env python3

import os
import sys
import time
import logging
import argparse
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
from utils_v4_4_0 import (
    setup_logging,
    save_pickle,
    parse_loci,
    filter_missingness,
    get_classifier,
    auto_select_classifier,
    tune_classifier,
    run_bootstrap,
    average_report,
    summarize_non_loci_columns,
    generate_roc_curves,
    generate_precision_recall_curve,
    generate_calibration_plot,
    predict_with_bootstrap,
    add_per_class_summaries,
    summarize_metadata,
    plot_prediction_probabilities,
    compute_locus_metrics,
    resample_data,
    plot_stacked_heatmap,
    phylogeny_impute,
    one_in_one_out_cv,
    generate_allelic_matrix,
    compartmentalized_predictions
)

# Generic warnings filtering
import warnings
warnings.filterwarnings("ignore", message=".*No further splits with positive gain.*")
warnings.filterwarnings("ignore", message=".*Auto-choosing.*")
warnings.filterwarnings("ignore", message=".*TBB Warning.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="SourceRunnerML v4_4_0")
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
    parser.add_argument('--mode', choices=['lite', 'advanced', 'pro'], default='lite')
    parser.add_argument('--run_name', required=True)
    parser.add_argument('--tune', action='store_true', help="Enable hyperparameter tuning (placeholder)")
    parser.add_argument('--metadata_summary', action='store_true', help="Generate metadata summary output")
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()

def apply_mode_settings(args):
    if args.mode == 'lite':
        args.bootstrap = 10
        args.resample_method = "none"
    elif args.mode == 'advanced':
        pass
    elif args.mode == 'pro':
        pass
    logging.info(f"Mode set to: {args.mode}")
    return args

def main():
    try:
        start = time.time()
        args = parse_args()
        args = apply_mode_settings(args)
        if args.pred_bootstrap is None:
            args.pred_bootstrap = args.bootstrap
        args.output_dir = f"outputs/{args.run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(args.output_dir, exist_ok=True)
        setup_logging(args)
        
        # Log input arguments
        logging.info("Input Arguments:")
        for arg, value in vars(args).items():
            logging.info(f"  {arg}: {value}")
        
        # Load training data
        train_df = pd.read_csv(args.train_file, sep='\t', dtype=str, low_memory=False)
        if args.loci_pattern:
            args.loci_prefix = ""
        loci = parse_loci(args, train_df)
        loci = filter_missingness(train_df, loci, args.missingness)
        logging.info(f"Proceeding with {len(loci)} loci after missingness filtering")
        
        # Reload and process training data
        train_df = pd.read_csv(args.train_file, sep='\t', dtype=str, low_memory=False)
        train_df[loci] = train_df[loci].apply(pd.to_numeric, errors='coerce')
        if args.missing_values == 'impute':
            train_df[loci] = SimpleImputer(strategy='most_frequent').fit_transform(train_df[loci])
        else:
            train_df = train_df.dropna(subset=loci)
        
        # Optionally apply phylogeny-informed imputation in advanced/pro modes
        if args.mode in ['advanced', 'pro']:
            train_df[loci] = phylogeny_impute(train_df[loci])
        
        X = train_df[loci]
        y = train_df[[c for c in train_df.columns if c.lower() == 'source'][0]]
        logging.info("Training matrix loaded successfully.")
        
        # Optional resampling
        if args.resample_method != "none":
            X, y = resample_data(X, y, method=args.resample_method)
            logging.info(f"Resampled training data using method: {args.resample_method}")
        
        # Compute per-locus metrics and generate allelic matrix for Fst estimation
        locus_metrics = compute_locus_metrics(X, y)
        locus_metrics.to_csv(f"{args.output_dir}/{args.run_name}_locus_metrics.tsv", sep="\t", index=False)
        allelic_matrix = generate_allelic_matrix(X)
        allelic_matrix.to_csv(f"{args.output_dir}/{args.run_name}_allelic_matrix.tsv", sep="\t", index=False)
        
        # Classifier selection and optional tuning
        if args.auto_classifier:
            clf_name, auto_score = auto_select_classifier(X, y)
            logging.info(f"Auto-selected classifier: {clf_name} with accuracy {auto_score:.4f}")
        else:
            clf_name = args.classifier
        if args.tune:
            logging.info("Tuning hyperparameters (placeholder)...")
            base_clf = get_classifier(clf_name)
            param_grid = {'n_estimators': [50, 100, 200]}
            tuned_clf = tune_classifier(base_clf, param_grid, X, y)
            tuned_clf.fit(X, y)
            final_model = tuned_clf
            clf_name = clf_name + "_tuned"
        else:
            final_model = get_classifier(clf_name)
            final_model.fit(X, y)
        
        # Bootstrapping for self-test metrics
        if args.bootstrap:
            logging.info(f"Running {args.bootstrap} bootstraps (burn-in: {args.burn_in}) on {args.cpus} CPU(s)")
            reports, cms, best_model = run_bootstrap(X, y, loci, clf_name, args)
            report_summary = average_report(reports, burn_in=args.burn_in)
            overall_accuracy = report_summary.get("accuracy", None)
            f1_scores = {cls: report_summary[cls]["f1-score"] for cls in final_model.classes_ if cls in report_summary}
            avg_cm = np.mean(cms, axis=0)
        else:
            report_summary = classification_report(y, final_model.predict(X), output_dict=True)
            overall_accuracy = report_summary.get("accuracy", None)
            f1_scores = {cls: report_summary[cls]["f1-score"] for cls in final_model.classes_ if cls in report_summary}
            avg_cm = confusion_matrix(y, final_model.predict(X))
        
        # Retrain final model if requested
        if args.retrain:
            logging.info("Retraining final model on full training data.")
            final_model = get_classifier(clf_name)
            final_model.fit(X, y)
            save_pickle(final_model, f"{args.output_dir}/{args.run_name}_retrained_clf.p")
        
        # Evaluate final model on training data
        final_preds = final_model.predict(X)
        final_cm = confusion_matrix(y, final_preds)
        clf_report = classification_report(y, final_preds, output_dict=True)
        
        # Write detailed training report
        with open(f"{args.output_dir}/{args.run_name}_training_report.txt", "w") as f:
            f.write("SourceRunnerML v4_4_0\n")
            f.write("Input Arguments:\n")
            for arg, value in vars(args).items():
                f.write(f"  {arg}: {value}\n")
            f.write(f"\nTraining samples: {X.shape[0]}\n")
            f.write(f"Loci used: {X.shape[1]}\n")
            f.write(f"Classes: {final_model.classes_.tolist()}\n")
            if overall_accuracy is not None:
                f.write(f"\nSelf-test Accuracy (Bootstrap Average): {overall_accuracy:.4f}\n")
            if f1_scores:
                f.write("\nF1 Scores per Class:\n")
                for cls, score in f1_scores.items():
                    f.write(f"  {cls}: {score:.4f}\n")
            f.write("\nConfusion Matrix (Final Model on Training Data - Real vs Predicted):\n")
            f.write("Rows: Actual, Columns: Predicted\n")
            f.write(np.array2string(final_cm, separator=', '))
            f.write("\n\nDetailed Classification Report:\n")
            f.write(pd.DataFrame(clf_report).to_string())
        
        # Optional metadata summary
        if args.metadata_summary:
            metadata_summary = summarize_metadata(train_df)
            metadata_summary.to_csv(f"{args.output_dir}/{args.run_name}_metadata_summary.tsv", sep="\t")
        else:
            logging.info("Metadata summary not generated (flag --metadata_summary not set).")
        
        # Prediction block
        if args.predict_file:
            logging.info("Starting prediction block...")
            pred_df = pd.read_csv(args.predict_file, sep='\t', dtype=str, low_memory=False)
            metadata_cols = [col for col in pred_df.columns if not col.startswith('CAMP')]
            pred_df[loci] = pred_df[loci].apply(pd.to_numeric, errors='coerce')
            if args.missing_values == 'impute':
                pred_df[loci] = SimpleImputer(strategy='most_frequent').fit_transform(pred_df[loci])
            else:
                pred_df = pred_df.dropna(subset=loci)
            probs, preds, ci_low, ci_high = predict_with_bootstrap(pred_df[loci], final_model, args.pred_bootstrap, args.cpus)
            metadata_df = pred_df[metadata_cols].copy()
            metadata_df['Predicted_Source'] = preds
            metadata_df['Confidence'] = probs.max(axis=1)
            predicted_class_indices = np.argmax(probs, axis=1)
            ci_lower_final = [float(ci_low[i, idx]) for i, idx in enumerate(predicted_class_indices)]
            ci_upper_final = [float(ci_high[i, idx]) for i, idx in enumerate(predicted_class_indices)]
            metadata_df['CI_Lower'] = ci_lower_final
            metadata_df['CI_Upper'] = ci_upper_final
            for i, cls in enumerate(final_model.classes_):
                metadata_df[f"Prob_{cls}"] = probs[:, i]
            metadata_df.loc[metadata_df['Confidence'] < args.min_confidence, 'Predicted_Source'] = 'Uncertain'
            pred_out_path = f"{args.output_dir}/{args.run_name}_srml_out.tsv"
            logging.info(f"Writing predictions to {pred_out_path}")
            metadata_df.to_csv(pred_out_path, sep="\t", index=False)
            
            # Generate prediction summary with counts and percentages
            prediction_counts = metadata_df['Predicted_Source'].value_counts()
            total_preds = prediction_counts.sum()
            prediction_percentages = (prediction_counts / total_preds * 100).round(2)
            prob_columns = [f"Prob_{cls}" for cls in final_model.classes_]
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
            
            # If additional metadata columns exist (e.g., 'country'), generate group breakdown
            if 'country' in metadata_df.columns:
                country_breakdown = metadata_df.groupby('country')['Predicted_Source'].value_counts(normalize=True) * 100
                country_breakdown = country_breakdown.round(2)
                country_breakdown.to_csv(f"{args.output_dir}/{args.run_name}_prediction_country_breakdown.tsv", sep="\t")
            
            # Generate combined histogram for predicted probabilities
            plot_prediction_probabilities(probs, args.output_dir, final_model.classes_)
            # Generate overall heatmap for per-isolate predicted probability distribution
            plot_stacked_heatmap(probs, args.output_dir, final_model.classes_)
            # Generate compartmentalized predictions summary
            mean_preds, std_preds = compartmentalized_predictions(final_model, pred_df[loci], n_folds=5, cpus=args.cpus)
            comp_summary = pd.DataFrame({
                "Mean_Probability": np.mean(mean_preds, axis=1),
                "Std_Probability": np.mean(std_preds, axis=1)
            })
            comp_summary.to_csv(f"{args.output_dir}/{args.run_name}_compartmentalized_prediction_summary.tsv", sep="\t")
            
            # Optionally, generate additional plots (ROC, PR, Calibration) if needed.
            # generate_roc_curves(...) and generate_precision_recall_curve(...) can be called here.
            
        logging.info(f"✅ Run completed in {time.time() - start:.2f} sec")
    
    except Exception as e:
        logging.exception("Unhandled exception occurred:")
        raise

if __name__ == "__main__":
    main()
