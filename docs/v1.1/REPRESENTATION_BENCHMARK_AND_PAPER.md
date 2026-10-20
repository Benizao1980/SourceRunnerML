# Allele-representation benchmark and preliminary paper outline

Design only in this first grouped-validation PR. Do not recommend a new default
or invent benchmark performance. Preserve numeric mode for historical reproduction.

## Practical comparison

Use identical exported outer/inner partitions for every method and seed. Start
with deterministic synthetic clonal panels (source signals, shared lineages,
uneven sources, rare/unseen alleles, missing cells, absent sources) and later a
rights-checked, source-labelled public dataset. Never use frozen Peru inputs.

- Numeric historical baseline: current modal imputation + RF/logreg; explicitly
  disclose arbitrary ordinal interpretation and tie sensitivity.
- Nominal one-hot: training-fitted per-locus vocabularies, explicit missing token,
  sparse output, unknown categories ignored with an unknown-rate report; logreg
  and a compatible tree baseline. A label permutation maps columns, not geometry.
  Fit filtering/vocabulary only in inner training; never learn from queries.
- Equality/Hamming kNN and nearest-source baseline: unequal observed alleles cost
  one regardless of identifier. Define pairwise distance as unequal jointly
  observed loci / jointly observed loci; minimum overlap, no-overlap abstention,
  and denominator diagnostics are mandatory. Do not let missingness equal a match.
  Tune k only within training. Predictive vote scores are not calibrated risk.

Test bijective per-locus relabelling, unseen/missing alleles, all-missing loci,
source-free leakage columns and deterministic tie policies. Modal imputation is
not automatically label-invariant; missing-as-category or observed-only distance
is preferable for the new modes. Keep held-out rows fixed during all comparisons.

Resource ladder: 200 x 100 toy, 2,000 x 1,343, then bounded 6,500 x 1,343 synthetic
profiles. Record wall time, peak RSS, nonzero sparse entries/vocabulary size,
versions/threads/seeds and input hashes. Estimate one-hot size before allocation;
fail with a memory ceiling rather than densify. Chunk kNN distances (avoid N²
storage); cap k and query batches. Later increase only under an execution budget.
No realistic-size fits or biological benchmarks are run in this initial iteration.

## Contribution and competing approaches (primary sources inspected 2026-10-20)

| Approach | Actual task/input | Relationship to this software |
|---|---|---|
| [aiSource](https://github.com/narning1992/aiSource), [Arning et al. 2021](https://pubmed.ncbi.nlm.nih.gov/34662334/) | ML source prediction including cgMLST/WGS; XGBoost application | Direct predecessor/comparator. cgMLST+ML is not new. SourceRunnerML's candidate contribution is auditable reusable validation and reproducibility, not a novel classifier. |
| [sourceR / HaldDP](https://pmc.ncbi.nlm.nih.gov/articles/PMC5473572/) | Bayesian source attribution with subtype/source counts and epidemiological model assumptions | Different estimand/data requirements; classifier votes are not equivalent to sourceR infection fractions or posterior uncertainty. |
| [Asymmetric island / iSource](https://pmc.ncbi.nlm.nih.gov/articles/PMC2538567/) | Population-genetic MLST attribution accounting for mutation/recombination/migration | Mechanistic modelling alternative; no claim this discriminative wrapper replaces its assumptions. |
| [2023 methodology comparison](https://pmc.ncbi.nlm.nih.gov/articles/PMC10303420/) | ML, network and Bayesian approaches on cgMLST/k-mer inputs | Prior comparison already exists; a software resource must show reproducibility/validation utility, not claim first use. |
| [sourcerer/source-attribution-MLST](https://github.com/ncezid-biome/source-attribution-MLST) | Predict bacterial food sources from known-source MLST profiles | Additional reusable software comparator; audit exact versions/features before a final manuscript table. |
| [FEAST](https://www.nature.com/articles/s41592-019-0431-x) | Microbial-community mixture source tracking | Related terminology but different community-level input/estimand; not an interchangeable cgMLST classifier benchmark. |

This is a scoped comparison, not a systematic review or tested head-to-head
performance ranking. Some PMC direct opens encountered a browser challenge;
abstract/search records and the aiSource repository were accessible. Confirm
full methods and citations before manuscript claims. Benchmark datasets require
explicit source-label provenance and redistribution checks before acquisition.

## Three principal figures (planned; no invented results)

**Figure 1 — Workflow.** Allele matrix + exact-ID metadata/schema -> QC/locus list ->
validation split BEFORE fitting -> nested training-only preprocessing/encoding/
model selection -> held-out predictions -> calibrated/uncertain query outputs +
portable model + provenance. Show the independent-holdout boundary and separate
training from reporting; no genome inference/typing functionality is implied.

**Figure 2 — Validation.** Matched-partition stratified vs grouped/source-specific
performance and calibration; numeric/one-hot/Hamming comparisons; runtime/RSS
panel. Include per-source denominators and group-resampled intervals. Use only
measured, reproducible results; synthetic panels labelled functionality evidence.

**Figure 3 — Reproducible application.** An approved public known-source example,
query probability/abstention output and controlled reference-composition/absent-
source sensitivity. Separate classifier confidence, predictive accuracy and
source-population inference. Do not use principal findings of either Peru paper.

Editable SVG/PDF, white backgrounds, consistent typography, versioned plotting
code and source tables. No result figures are fabricated in this initial phase.

## Manuscript scope and release gate

Tentative title: *SourceRunnerML: a reproducible machine-learning framework for
genomic source attribution of bacterial pathogens*. Sections: need/prior work;
software; leakage-aware methods; benchmarks; worked example; limitations;
software/data availability. Keep geographic-reference and poultry-production
biological findings for their separate Peru studies. Confirm current Microbial
Genomics article-category instructions before manuscript submission. No release,
DOI/archive upload, merge or submission without review.
