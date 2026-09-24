# Salmonella pilot checklist

## Phase 1 — dataset
- [ ] Confirm public source dataset and redistribution terms.
- [ ] Select three well-represented source categories.
- [ ] Harmonise source labels and retain raw labels.
- [ ] Add serotype, geography, year and genomic cluster metadata where available.
- [ ] Document all inclusion/exclusion rules.

## Phase 2 — smoke test
- [ ] Confirm Salmonella cgMLST column parsing.
- [ ] Run preflight/QC on a small subset.
- [ ] Confirm end-to-end execution with RF and logistic regression.
- [ ] Check output probability vectors sum to 1.0.

## Phase 3 — pilot validation
- [ ] Compare RF, logistic regression and XGBoost.
- [ ] Run stratified 5-fold cross-validation.
- [ ] Run 25–50 bootstrap/OOB replicates.
- [ ] Add lineage-grouped cross-validation if cluster metadata are available.
- [ ] Add a geographic holdout if adequately powered.
- [ ] Inspect confusion matrices and class-specific performance.
- [ ] Evaluate probability calibration where feasible.

## Phase 4 — worked example
- [ ] Add reproducible commands.
- [ ] Add compact summary tables/figures.
- [ ] Document failure modes and limitations.
- [ ] State clearly that the pilot is not the final K01 dataset or analysis.

## Phase 5 — K01 linkage
- [ ] Update grant text only with results that have actually been generated.
- [ ] Distinguish existing SourceRunnerML functionality from features Erika will develop during the K01.
- [ ] Use the worked example as feasibility evidence, not as a substitute for the proposed research.
