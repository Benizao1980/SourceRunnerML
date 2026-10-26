# Optional grouped validation (v1.1 development)

Use declared, user-supplied genomic groups when related isolates must not cross
training/validation boundaries. SourceRunnerML does not infer ST, CC, LIN or
hierCC. Validate these metadata and the allele scheme before use.

`design.json`:

```json
{"strategy":"grouped","group_column":"lineage","id_column":"id","collision_policy":"error"}
```

```sh
python scripts/sourcerunner_full_validation.py \
  --train_file synthetic_train.tsv --predict_file synthetic_query.tsv \
  --source_col source --keep_sources A,B --loci_prefix CAMP \
  --models random_forest --cv_folds 3 --inner_cv_folds 2 \
  --validation_config design.json --cv_only \
  --output_dir results --run_name grouped --cpus 1 --random_seed 42
```

The existing `--tuning_config` can add training-only grouped inner grid search.
Every requested model uses the exact same outer partitions. Filtering/imputation
is fitted within each relevant training partition. Raw numeric and training-fitted
transformed-feature equality is audited at both outer and tuned inner boundaries.
Default collision policy fails. Explicit `warn` permits the overlap, prints a
warning even when generic sklearn warnings are suppressed, and records its size;
these scores must not be advertised as collision-free validation. Identical
profiles may genuinely occur in different groups; no groups are silently merged.
No group IDs, sample IDs or sources are permitted as selected feature columns.

## Deliberate scope boundary

**Grouped runs require `--cv_only`.** They stop after CV reports and model-family
comparison; no bootstrap model, final pickle or query attribution is produced.
Query input remains required for compatibility with existing locus-schema
selection; its allele values are not used to fit CV preprocessing.
The existing isolate-bootstrap path cannot be interpreted as lineage-aware
uncertainty. Whole-group bootstrap/inference is a separate release gate.

Omitting `--validation_config` retains the previous stratified workflow. Optional
`--cv_only` also permits a stratified comparator without bootstrap. Model-family
selection still uses the outer scores; a winning-family score is not an unbiased
independent assessment. No individual transmission or population-attributable
infection fraction follows from these classifier predictions.

## Feasibility and records

- Exact, nonblank unique sample IDs and nonblank groups are required. Each source
  needs at least as many distinct groups as requested folds; all sources must be
  present in train AND test in every outer/tuned inner fold.
- StratifiedGroupKFold is a heuristic; if its assignment fails support checks,
  request fewer folds or redesign explicitly. Failure does not prove no possible
  valid assignment exists. There is no silent downgrade to isolate CV.
- Isolate debug subsampling (`--max_train_rows`) is disallowed for grouped mode;
  select whole groups upstream and record that selection.
- `validation_fold_membership.tsv` records actual sample/group/source membership
  and roles for each fold. It is an audit export; importing user-defined manifests
  is planned, not implemented here.
- `validation_manifest.json` records configuration, seed, outer source counts,
  raw collision summaries and input/configuration/locus/tuning SHA-256 hashes.
  `validation_exclusions.tsv` records source-filter exclusions. Retained loci and
  transformed collision checks are in per-model fold reports; inner group
  memberships, seed and collision checks are recorded within tuning JSON fields.
- `cv_oof_predictions__MODEL.tsv` exports sample IDs (or row positions for the
  default stratified path), true/predicted source, fold and probability columns.
  These are classifier probabilities, without an automatic calibration claim.

Output manifests contain sample identifiers; filtered training rows are also
exported by the existing wrapper. Keep real outputs controlled and publish only
rights-cleared benchmark inputs/aggregates. Numeric allele mode remains unchanged;
nominal encodings and Hamming comparisons are planned separately. Missing values
continue through the existing training-fitted modal imputation path.

Synthetic tests: `python -m unittest discover -s tests -p 'test_grouped_validation.py' -v`.
No protected data, research-result replacement, inference or HPC execution occurs.
