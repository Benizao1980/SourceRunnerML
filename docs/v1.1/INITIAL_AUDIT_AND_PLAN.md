# SourceRunnerML v1.1: initial audit and release-candidate plan

Audit date: 2026-10-20. Default `main`: `735fe57349ff750e4f34c85e6cb1ea3a21868f22`.
Source inspection only before implementation; no empirical analyses. The public
repository has two open issues (#11, #17), no open PRs, and no GitHub releases.
No repository AGENTS.md exists at this commit. Portfolio safeguards apply.

## Already implemented, verified against source and merged PRs

- **#15** (`c8623cc`): training-partition missing-locus filtering and modal
  imputation inside sklearn pipelines; outer CV, bootstrap in-bag and final-fit
  paths share this transformer. Explicit locus files, regex and legacy CAMP
  prefixes are supported. These fixes must not be described as outstanding.
- **#16** (`fe1503a`): optional bounded parameter grids, training-only inner
  StratifiedKFold, parameter/CPU/seed safeguards and tuning provenance. Model
  families are still selected on the same outer CV whose score is reported.
- **#18** (`735fe57`): portable model/imputer/locus/label-encoder pickle dictionaries;
  downstream loading does not require importing custom project transformers.
- Supported wrapper: RF, logistic regression and optional XGBoost/LightGBM/CatBoost;
  balanced accuracy, macro/weighted F1, source precision/recall, raw confusion
  matrices, isolate bootstrap/OOB metrics, replicate probability mean/SD,
  entropy, consensus and a probability-threshold `Uncertain` label.
- README claims OOF predictions, but source writes aggregated CV reports and
  internally accumulated hard labels, not a reusable per-isolate OOF probability
  table. Calibration scores/plots, group-aware intervals and independent-holdout
  orchestration are missing.

## Remaining correctness and reproducibility risks

Generic outer and inner validation is isolate-stratified; clones/geographies
can cross boundaries. Bootstrap sampling is also isolate-based. Duplicated in-bag
isolates can cross inner tuning folds, even though outer OOB data are withheld.
Exact/raw and transformed-feature collisions are not audited. Numeric allele IDs
can impose arbitrary thresholds or distances. Modal-imputation tie breaking may
also depend on allele numbering. Missing/multiallelic tokens become NaN; arbitrary
nonnumeric allele strings are coerced to missing, so representation comparisons
must explicitly retain nominal labels in their new input path.

Legacy `SourceRunnerML.py` / `_v1_0.py` retain global preprocessing and optional
resampling before self-test splitting; do not rewrite frozen historical analyses.
Issue #17 correctly separates historical reanalysis from the completed #15 fix.
Issue #11 is research-owner C. coli work, outside this software iteration.

No installed package/pyproject or citation metadata; dependencies are unpinned
minimal/full lists, optional imports may fail per model, warnings are broadly
suppressed, timestamp output directories can collide, full filtered training
rows are exported, and pickle requires trusted-origin loading. The installed scikit-learn 1.9.1 also warns that LogisticRegression.n_jobs will
be removed in 1.10; resolve/version-test that dependency before release. CI exercises
synthetic generator, preprocessing, tuning and portable loading on Python 3.11;
path filters need the new module/tests. No supported empirical Salmonella result
is established by synthetic locus-renaming smoke tests.

## Smallest defensible publishable v1.1 set (release gates, not completed claims)

1. Group-contained outer/inner validation, reproducible partition manifests,
   collision/class feasibility checks and clearly scoped resampling uncertainty.
2. Explicit fold manifests and verified geographic/time holdouts; independent
   evaluation of model-family selection, with no future-to-past fitting.
3. Numeric historical mode plus nominal one-hot and equality/Hamming baselines,
   compared on identical frozen partitions with missing/unseen allele tests.
4. OOF/holdout probabilities, balanced/source metrics, denominated confusion,
   calibration diagnostics, group-resampled uncertainty, abstention and absent-
   source sensitivity. Distinguish confidence from epidemiological attribution.
5. Synthetic worked example plus an approved redistributable empirical benchmark,
   pinned clean installation, portable loading, release notes/citation/archive
   proposal, and the focused Technical Resource manuscript.

No first-author biological claims from either Peru study belong in this paper.
The release candidate is blocked on empirical input rights/design review and
these validation gates; this iteration must not label it release-ready.

## First independently reviewable implementation

Add optional grouped configuration to the existing wrapper. Precompute the same
outer folds for all models, fit preprocessing inside every training fold, enforce
training-only grouped inner tuning, and write memberships/hashes/seeds/class counts.
Require `--cv_only` for grouped runs: whole-group bootstrap uncertainty needs a
separate sampling/estimand review and is deliberately deferred, not replaced by
the existing isolate bootstrap. The unchanged default still supports its historical
isolate-bootstrap inference path. Reject missing group/ID values,
class-incomplete partitions and leakage collisions with actionable errors.
Keep unconfigured isolate-stratified behaviour reproducible. Do not add all future
strategies or refactor unrelated legacy code in this PR.

## Planned API

`--validation_config design.json`, e.g.

```json
{"strategy":"grouped","group_column":"clonal_complex","id_column":"id","collision_policy":"error"}
```

The current `--cv_folds`, `--inner_cv_folds` and `--random_seed` remain authoritative.
ST/CC/LIN/hierCC labels must be supplied by the user; never infer lineages from
source labels. Debug isolate subsampling is disallowed for grouped runs.

Subsequent configs, **not implemented in the first PR**:

- `manifest`: long table `(fold, role, sample_id)` with explicit train/test/exclude
  roles, exact ID joins, every declared group kept intact and disjoint roles.
- `geographic`: verified metadata column and held-out values; unknown geography
  excluded with recorded decisions, cross-border genomic groups never split.
- `temporal`: verified sampling dates, explicit cutoff(s) and optional gap; max
  training date strictly precedes minimum test date. Groups spanning time boundary
  must be explicitly excluded or placed wholly in a scientifically valid role.

All strategies require class support checks at outer AND inner boundaries.
Impossible designs fail; a splitter heuristic failing is not proof no feasible
assignment exists. Never silently relax group containment.

## PhyloCloud and portfolio integration

PhyloCloud stays P1. PRs #23-25 are marked merged into their respective bases;
#26 remains draft/open. Main previously contained only #23; preserve the stacked
benchmark/composition/detail history and publication figures. Outstanding P1:
zoom-aware density, visual acceptance and approved-real-tree validation. Do not
merge the stack or interrupt other agents/runs. SourceRunnerML is a separate
additional workstream, explicitly authorized on 2026-10-20. No HPC/Peru data work.

## Initial verification (2026-10-20)

15 synthetic unittest tests passed, including grouped nested-fit spies, outer
held-out perturbation, raw/transformed collision checks, feasibility/ID checks
and a subprocess grouped-CV run that writes memberships/OOF probabilities and
refuses the isolate-bootstrap continuation. A separate conventional RF/logreg
90-training/12-query toy run with two CV folds and two bootstrap replicates
completed; probabilities are functionality checks, not biological results.
CI additionally retains its previous synthetic locus-scheme and pickle round-trip
checks. Other optional classifiers and realistic-size representation benchmarks
were not fitted locally. Current environment is recorded separately.
