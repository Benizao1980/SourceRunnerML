# Salmonella source attribution — Erika Austhof / K01 example strand

**Status: proposed collaborative study and training pathway (October 2026). No Salmonella attribution results are provided here.** This directory is a distinct scientific strand connected to the [general Salmonella feasibility pilot](../salmonella_pilot/); it is not a claim that Erika's K01 data or a validated Salmonella model are publicly available.

## Biological/public-health objective

Extend SourceRunnerML from *Campylobacter* to *Salmonella enterica* and test whether population-genomic patterns in source-labelled references can inform assignment of human isolates to likely **represented source populations**, while accounting for geography, clonal/serotype structure and epidemiological context.

This is relevant to Erika's K01 source-attribution aim. The wider programme includes weather/climate, eBird/migration, Salmonella genomic population structure and FoodNet-linked epidemiology; this strand addresses the **genomic attribution** question and its validation, not a claim that weather directly causes an individual infection.

## Relationship to other repository examples

| Example | Purpose | Status |
|---|---|---|
| [Toy tutorial](../../docs/tutorial.md) | Learn inputs, QC, classifier outputs with synthetic profiles | Runnable educational demo |
| [Peru geography](../peru_geographic_context/START_HERE.md) | Real-world reference-geography and leakage/lineage validation lesson | C. jejuni analysis documented; requires archived inputs/scripts to reproduce |
| [Generic Salmonella pilot](../salmonella_pilot/) | Test organism adaptation, source harmonisation, locus selection | Feasibility scaffold; no results |
| **Erika / K01 Salmonella (this page)** | Study-specific design connecting genomic sources to human/public-health context | Proposed; no private data committed |

## Core research questions

1. What source categories are scientifically credible and adequately represented in the available EnteroBase/public or permitted reference collections?
2. Can a model discriminate these sources when lineage/serotype relatedness is controlled, rather than merely memorising common clones?
3. Does performance generalise across states, collection years, Salmonella serotypes and hierCC clusters?
4. How does reference geography alter model performance and uncertainty, analogous to the question asked in Peru?
5. Among human isolates with linked FoodNet epidemiology, do predicted source associations agree with independently defined exposure or risk histories, and where do they disagree?

**Separate tasks:** modelling source population similarity; describing human isolate populations; validating against exposure histories. Exposure histories are not source ground truth and should not be used to circularly train/evaluate attribution.

## Candidate data layers — availability to verify

| Data layer | Potential use | Access/validation requirement |
|---|---|---|
| EnteroBase/public Salmonella genomes + cgMLST alleles | Source-labelled training and genomic distance/cluster information | Verify export permissions, sample IDs, isolate quality and source metadata |
| Source metadata | Define poultry, cattle/ruminant, wild-bird and other classes if supported | Harmonise without overstating reservoir specificity |
| Salmonella serotype + EnteroBase hierCC | Stratification, outbreak/lineage-aware splits and sensitivity analysis | Identify suitable hierCC resolution empirically; no LIN codes |
| State/region + collection date | Geographic/temporal holdouts and overlap audits | Verify metadata completeness and representativeness |
| Human Salmonella genomes + FoodNet epidemiology | Unlabelled targets and independent comparison with exposure hypotheses | Controlled access and governance; do not upload records to public GitHub |
| eBird / migration / weather outputs | Wider K01 epidemiological context, separate secondary association research | Do not feed outcome-derived epidemiological signals into genomic-only source model |

Earlier project planning has discussed Colorado, New Mexico, Utah, Alabama, Tennessee and Georgia, but **do not treat these as confirmed final sampling sites or dataset strata** without checking the current K01 protocol.

## Phase A — define a defensible pilot

- Start with a small number of candidate labels (illustratively Poultry, Ruminant, Wild bird) **only if** each has sufficient verified reference diversity; change categories after metadata audit.
- As a feasibility target, seek hundreds of isolates/class where permissible; the nominal 300–500/class target in the generic pilot is not an observed sample count.
- Predefine exclusion of ambiguous, environmental or multiply sourced records rather than assigning them opportunistically.
- Audit duplicated and near-identical genomes, serotype, hierCC cluster, state, year and source composition.
- Use the *same* Salmonella allele scheme for train/test and query isolates.

## Phase B — make SourceRunnerML accept the export

SourceRunnerML v1.0's full-validation wrapper expects a **single genomic-locus prefix**, currently CAMP by default; this is not a suitable assumption for arbitrary EnteroBase exports. Track and resolve [issue #5](https://github.com/Benizao1980/SourceRunnerML/issues/5) with an explicit locus list or verified prefix before running real Salmonella data.

Until then, the generic Salmonella [run_pilot.sh](../salmonella_pilot/scripts/run_pilot.sh) intentionally stops if no compatible locus prefix is supplied. **Do not rename columns blindly**: keep a stable mapping to the original Salmonella locus identifiers.

A preliminary input table would contain:

~~~text
id  raw_source  harmonized_source  serotype  hierCC  state  year  <Salmonella cgMLST locus columns...>
~~~

and the human prediction table would contain the same locus set plus permissible, non-public ID metadata. The source label is absent from the query set.

## Phase C — methods and validation

1. **Preflight:** source-label audit, loci shared/filtered, completeness, isolate deduplication, geography and serotype composition.
2. **Naive baseline:** random forest and multinomial logistic regression, adding boosted trees only when dependencies are stable.
3. **Lineage-aware validation:** separate held-out hierCC clusters (resolution selected and justified for Salmonella biology). Quantify the difference from isolate-random CV.
4. **Serotype-aware validation:** assess whether classifier is learning common serotypes rather than source-association within serotypes; stratify or hold out suitable serotypes.
5. **Spatial/temporal generalisation:** leave-one-state/region-out and forward-in-time testing where source counts allow.
6. **Reference-panel sensitivity:** broader US/public panel versus local/region-matched panel, with class- and sample-size-matched analyses. Avoid overlap by IDs and genomic profiles.
7. **Probabilities:** report class-specific recall, balanced accuracy, macro-F1, confusion, calibration curves and uncertainty; avoid presenting arbitrary thresholded probabilities as calibrated.
8. **Independent check:** compare human probability distributions with linked FoodNet exposure variables, accounting for selection bias and uncertain self-reporting.

**Important current limitation:** the standard full-validation wrapper currently performs preprocessing before stratified CV/bootstrap evaluation, making leakage possible. A publication analysis should use a verified fold-contained preprocessing pipeline and prespecified grouped folds. See [the tutorial warning](../../docs/tutorial.md#8-critical-scientific-cautions).

## First practical deliverables

- [ ] Approved data-access/redistribution plan and input inventory
- [ ] Human/source separation and metadata dictionary with source/serotype/hierCC/geography/year counts
- [ ] Confirmed Salmonella cgMLST scheme and versioned list of locus names
- [ ] Source harmonisation table with raw and harmonised values
- [ ] Deduplication, overlap and leakage report
- [ ] Pilot baseline and grouped/external validation scripts, with fixed seeds
- [ ] Non-sensitive aggregate class-specific results with limitations
- [ ] Study-specific reproducibility bundle suitable for Erika's K01 methodology/training record

## Scientific terms to introduce when teaching

A **reservoir** can maintain and circulate bacterial populations; an **exposure vehicle** may be food or environment; a **source label** in a reference database describes where an isolate was sampled. These are related but are **not identical causal concepts**.

A **hierCC cluster** is an EnteroBase hierarchical cgMLST grouping, useful for lineage-aware comparisons; it should not be assumed equivalent to a Campylobacter LIN code. **Serotypes**, **sequence types**, **hierCC groups** and **individual isolates** resolve population structure at different scales. Split choice can materially change estimated generalisation.

## Governance and reproducibility

Do not commit restricted FoodNet records, identifiable patient metadata, controlled-access genomes, private K01 material or raw contact details. Share analysis-ready synthetic data, permissible public accessions, aggregate results and scripts. The precise datasets, participation and contribution statements should be confirmed with collaborators before publication.

## Links

- [General Salmonella feasibility plan](../salmonella_pilot/PLAN.md)
- [SourceRunnerML full getting-started guide](../../docs/getting_started.md)
- [Teaching tutorial and glossary](../../docs/tutorial.md)
- [Peru example — why geography matters](../peru_geographic_context/START_HERE.md)
- [Erika Austhof's RainDroughtSalmonella compendium](https://github.com/austhofe/RainDroughtSalmonella) (separate project; not a SourceRunnerML data source by default)
