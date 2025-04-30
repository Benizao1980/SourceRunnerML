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

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
