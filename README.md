# SourceRunnerML v0_4_108
SourceRunnerML is an advanced machine learning pipeline for microbial source attribution using (cg)MLST data. It combines robust model training with putative source prediction while accommodating a wide range of input schemes and multiple enteric species. The pipeline leverages bootstrapping, parallel processing, and advanced resampling methods to improve model performance and generate reliable predictions.

## ✨ New in v0_4_108
- Optional `Plotly`-based output visualizations
- Enhanced prediction outputs:
  - Per-sample probability breakdown
  - Confidence intervals
  - ROC/AUC metrics
- Improved classifier handling (e.g., XGBoost, LightGBM, CatBoost)
- Smoother handling of missing data and mixed-type locus values

## Usage
Run the main script from the command line. For example:

```bash
python3 SourceRunnerML_v0_4_108.py \
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
    --run_name SR-ML_v0_4_108 \
    --metadata_summary \
    --verbose


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
## 🔍 Features Explained

SourceRunnerML provides a modular and extensible platform for source attribution using (cg)MLST data. Below is a breakdown of the key components:

### 📈 Model Training & Prediction
- **Combined Training & Prediction Workflow**: Train on one dataset and predict on another in a single streamlined run.
- **Auto Classifier Selection (`--auto_classifier`)**: Automatically tests Random Forest, Logistic Regression, XGBoost, LightGBM, and CatBoost to choose the best-performing model.
- **Ensemble Support**: Soft-voting ensemble classifier across top models to improve robustness.

### ⚙️ Data Preprocessing & Handling
- **Generic Locus Detection**: Automatically detects locus columns based on a user-defined prefix (e.g., `CAMP`) or regex.
- **Missing Data Handling**:
  - Drop or impute missing values (default: most frequent allele).
  - Control missingness threshold via `--missingness`.
- **Smart Imputation (Planned)**: Placeholder for ST- and phylogeny-informed imputation strategies.

### 🔄 Resampling & Balancing
- **Class Imbalance Handling**:
  - Supports `undersample`, `oversample`, and `SMOTE`.
  - Automatically selects best method based on training performance.
- **Bootstrap & Burn-In**:
  - Bootstrap models with defined burn-in iterations to improve variance estimation.

### 🧪 Evaluation & Diagnostics
- **Self-Test Accuracy**: Report internal training accuracy, ROC, AUC, and precision-recall.
- **Cross-Verification**: Compartmentalizes predictions across bootstrapped subsamples.
- **Per-Locus Metrics**: Outputs statistics like Fst (placeholder), frequency, and missingness per locus.

### 📊 Visualizations & Reporting
- **Prediction Output Includes**:
  - Top predicted source
  - Per-sample probability breakdown
  - Confidence intervals for predicted class
  - Per-class breakdown table
- **Plotly Visuals** (optional):
  - Interactive bar plots of prediction confidence
  - Summary heatmaps and PCA (planned)
- **Metadata Summary**: Optionally includes training/prediction metadata breakdowns in the report.
- **Export Support**:
  - Microreact- and iTOL-compatible files for phylogenetic coloring and visualization.

---
## 🚧 SourceRunnerML Development Roadmap
### ✅ Completed
-  Added LightGBM & CatBoost classifiers, ensemble methods, and generic locus detection.
- Enhanced reports with per-locus metrics, metadata summaries, and group-wise evaluations.
- Introduced resampling (undersample, oversample, SMOTE), burn-in training, and auto-selection of resampling methods.
- CLI overhaul with support for `--mode`, config files, Microreact/iTOL export, and placeholders for phylogeny-informed imputation.
- Added diagnostic mode, one-in-one-out cross-verification, refined uncertainty handling, and run diagnostics.

---
### 🧭 Immediate Priorities
#### Advanced Data Imputation
- MLST- or phylogeny-informed imputation
- Tajima’s D and F-statistics in locus-metrics output

#### Uncertainty & Error Propagation
- One-in-one-out CV during prediction
- Propagate training error into prediction CIs

#### Model Explainability
- SHAP value integration
- Feature-importance summary and heatmap

---
### ⏱️ Short-Term Goals
#### Visualization Enhancements
- Per-class ROC and PR curves
- 100% stacked bar by predicted source
- PCA of isolate clustering (train vs predict)
- Heatmap of source probabilities by sample
- Color palette customisation

#### CLI Refinement
- Defined modes: `default`, `diagnostic`, `pro`
- Flags to toggle locus metrics, visuals, and metadata

---
### 🧬 Mid-Term Goals
#### Integration & Distribution
- Streamlit GUI for point-and-click runs
- Snakemake/Nextflow support for HPC workflows
- PyPI and Conda distribution with versioned changelogs

#### Pretrained Models
- Prebuilt models for key pathogens (e.g., *Campylobacter*, *Salmonella*)
- Ability to skip training using existing models

---
### 🌐 Long-Term Vision
#### Assembly-Based Source Attribution
- Accept raw genomes and derive k-mers or unitigs

#### Enhanced Self-Testing
- Randomized isolate selection for bootstraps
- Multi-fold cross-verification

#### Continual Learning
- Online updates from new isolate input

#### Advanced Ensemble Learning
- Meta-learning with stacking and blending methods

---
## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
