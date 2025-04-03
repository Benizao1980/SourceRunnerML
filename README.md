# SourceRunnerML

SourceRunnerML is an advanced machine learning pipeline for microbial source attribution using (cg)MLST data. It combines robust model training with putative source prediction while accommodating a wide range of input schemes and multiple enteric species. The pipeline leverages bootstrapping, parallel processing, and advanced resampling methods to improve model performance and generate reliable predictions.

## Key Features

- **Combined Model Training & Putative Source Prediction**  
  Train a model on high-dimensional (cg)MLST allele data and predict the most likely source (e.g., chicken, wild bird, ruminant, etc.) for each isolate—all within a single streamlined workflow.

- **Flexible Input for Multiple (cg)MLST Schemes**  
  Automatically detects loci using a specified prefix or regex. Designed to support various (cg)MLST schemes with plans to extend support to multiple enteric species (e.g., Salmonella, E. coli).

- **Bootstrapping for Improved Model Performance**  
  Uses bootstrapping during training to generate self-test metrics and estimate model variance, enabling the calculation of robust confidence intervals and performance metrics.

- **Multiple ML Classifiers & Model Tuning (Under Development)**  
  Supports several classifiers (RandomForest, Logistic Regression, XGBoost, LightGBM, CatBoost, and ensemble methods) to identify the best-performing model. Hyperparameter tuning is enabled via GridSearchCV (with plans for Bayesian optimization).

- **Flexible Missing Data Handling**  
  Offers two approaches:
  - **Missingness Threshold:** Excludes loci with a high fraction of missing data (set via `--missingness`).
  - **Missing Data Options:** Choose to impute missing values (default: impute using the most frequent allele) or drop rows with missing data using the `--missing_values` flag.

- **Resampling Options for Class Imbalance**  
  Addresses class imbalance via:
  - Undersampling  
  - Oversampling  
  - SMOTE (Synthetic Minority Oversampling Technique)  
  Select the method with the `--resample_method` flag.

- **Parallelisation for Improved Performance**  
  Leverages multiple CPU cores during bootstrapping and prediction, reducing runtime on large datasets.

- **Compartmentalisation of Prediction Data for Cross Verification**  
  Splits the prediction dataset into folds, runs predictions on each compartment, and aggregates results. This approach provides robust confidence intervals and cross-verification of predictions.

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Benizao1980/SourceRunnerML.git
   cd SourceRunnerML
   ```

2. **Create a Virtual Environment (Recommended):**

   Using Conda:

   ```bash
   conda create -n srml python=3.8
   conda activate srml
   ```

   Or using venv:

   ```bash
   python3 -m venv srml-env
   source srml-env/bin/activate
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main script from the command line. For example:

```bash
python3 SourceRunnerML_v4_4_0.py \
    --train_file path/to/training_isolates.txt \
    --predict_file path/to/prediction_isolates.txt \
    --loci_prefix CAMP \
    --missing_values impute \
    --missingness 0.2 \
    --auto_classifier \
    --bootstrap 20 \
    --pred_bootstrap 20 \
    --burn_in 5 \
    --cpus 8 \
    --resample_method smote \
    --retrain \
    --min_confidence 0.5 \
    --mode advanced \
    --run_name SR-ML_v4_4_0 \
    --metadata_summary \
    --verbose
```

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your changes, and submit a pull request. Be sure to update documentation and tests as necessary.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
