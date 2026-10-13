# Peru geographic-context study — start here

This is a **research reproducibility guide**, not a one-command downloadable demonstration. The completed *C. jejuni* manuscript analysis uses private/controlled or separately stored inputs and a fixed analysis design that is **not implemented in the generic SourceRunnerML v1 wrapper**.

If you only want to learn how to use the software, complete the [toy tutorial](../../docs/tutorial.md) first.

## The question in one sentence

Does including geographically matched source genomes improve source attribution to Peruvian human *Campylobacter jejuni* relative to non-Peru reference populations, and can this improvement be explained solely by sample size?

See [README.md](README.md) for the frozen results and [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the complete methodology.

## Study populations and comparisons

| Component | Frozen or planned status |
|---|---|
| Known-source Peru *C. jejuni* | 1,076 poultry + 149 ruminant for the 1,225-isolate formal benchmark |
| Excluded from formal Peru benchmark | 11 Peru pig isolates (too few to estimate a robust class metric) |
| Global source panel | 17,372 original; 127 Peru records removed; 17,245 remaining non-Peru |
| Human Peru *C. jejuni* | 981 prediction isolates |
| Reference strategies | Non-Peru global; Peru-local out-of-fold; combined global + local out-of-fold |
| Main benchmark | Fixed fivefold genomic-lineage-blocked comparison across the *same* held-out isolates |
| *C. coli* | Input reconstruction and evaluation design still pending |

Frozen known-source benchmark:

| Reference panel | Balanced accuracy | Ruminant recall |
|---|---:|---:|
| Non-Peru global | 0.754 | 0.510 |
| Peru-local OOF | 0.875 | 0.758 |
| Combined OOF | **0.893** | **0.792** |

These values are **documented outcomes of the prior analysis**, not outputs automatically reproduced by installing this repository.

The underlying aggregate confusion counts are now in [`results/cjejuni_known_source_benchmark.tsv`](results/cjejuni_known_source_benchmark.tsv). To check their arithmetic without any private data, from the repository root run:

~~~bash
python examples/peru_geographic_context/scripts/verify_benchmark_summary.py
~~~

This is an **aggregate consistency check, not a model rerun**. See [`results/README.md`](results/README.md).

## What is actually needed to reproduce the paper?

You need the source-labelled *C. jejuni* global and local allele matrices, the Peru human matrix, harmonised metadata, leakage-exclusion manifest, frozen genomic group/fold assignments, exact model versions and seeds, retained-locus manifest, plus the original study-specific scripts or a verified reimplementation. Large, non-redistributable inputs are intentionally absent from GitHub.

**Stop point:** do not describe the analysis as computationally reproducible from a fresh clone until those materials are made available or a public-source reconstruction protocol and script are checked end-to-end.

## How to inspect your own Peru inputs (without claiming a reproduction)

From the repository root, substitute the real paths:

~~~bash
python scripts/source_runner_preflight.py \
  --train /path/to/peru_reference_source.tsv \
  --predict /path/to/peru_human_cjejuni.tsv \
  --source_col source \
  --id_col id \
  --loci_pattern '^CAMP' \
  --min_source_size 20 \
  --outdir peru_preflight \
  --write_tsv
~~~

Check the report for the expected locus scheme, duplicated IDs, source counts, and training/prediction overlap using your separate data provenance checks. The preflight program does not prove genomic non-overlap or audit all aliases.

The generic wrapper can be used for an **exploratory, separate** pilot on those tables, but its stratified CV is **not** the fixed lineage-blocked three-reference benchmark. Do not put its metrics into the study's frozen results table.

## Reproduction checklist

1. **Provenance:** identify the exact version of each global, Peru-local and human export; record access conditions.
2. **Species separation:** fit distinct *C. jejuni* and *C. coli* analyses; do not combine loci across species.
3. **Leakage audit:** remove Peru-origin isolates from global references, match isolate aliases and screen genomic duplicates.
4. **Source-label harmonisation:** preserve raw labels and every exclusion; keep the 11 *C. jejuni* pigs out of the formal 2-class local benchmark.
5. **Fold manifest:** re-use the five frozen lineage-blocked folds; the same test isolates must be evaluated across reference strategies.
6. **Fold-contained training:** filter loci, impute, bootstrap and fit models using training partitions only; ensure no held-out lineage occurs in that fold's local training data.
7. **Evaluate:** calculate balanced accuracy, macro-F1, source recalls and confusion matrices on held-out, known-source isolates; report fold variability.
8. **Sensitivity:** re-run size-matched reference comparisons, learning curves and independent nearest-source checks.
9. **Human predictions:** fit the specified strategy, preserve full source probability vectors and apply the 0.60 reporting threshold separately.
10. **Archive:** model seeds/settings, Python package versions, fold/locus/exclusion manifests, prediction output, and tree QC.

## Why the result matters

Geographic context changes which source-associated lineages the model has seen. The combined panel retained broad global diversity while improving recognition of local ruminant-associated isolates. This is **not** a universal claim that local panels outperform global ones.

## C. coli: next decision, not a reported result

Reconstruct the complete source and human *C. coli* cgMLST matrices; inventory local class counts and ST-828/ST-1150; remove geographic/reference leakage; decide whether there is adequate power for a parallel lineage-blocked benchmark. If not, use lineage-resolved predictions and nearest-source/phylogenetic checks instead of forcing an unstable class comparison.

See [STATUS_AND_NEXT_STEPS.md](STATUS_AND_NEXT_STEPS.md) and [issue #11](https://github.com/Benizao1980/SourceRunnerML/issues/11).

## Reader map

- [Study results and biological interpretation](README.md)
- [Full method and frozen settings](REPRODUCIBILITY.md)
- [Progress, missing inputs and C. coli decision tree](STATUS_AND_NEXT_STEPS.md)
- [General SourceRunnerML tutorial](../../docs/tutorial.md)
