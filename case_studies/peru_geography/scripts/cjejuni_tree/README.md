# C. jejuni Figure 1 phylogeny workflow

This directory records the tested Prokka → PIRATE → IQ-TREE workflow used for the Peru *C. jejuni* Figure 1 phylogeny, including the recovery steps required after alignment-QC and memory failures.

## Final production result

- initial population: **2,217 Peru C. jejuni genomes**
- PIRATE core alignment: **1,405,341 bp**
- taxon QC threshold: exclude taxa with **>50% missing/ambiguous core-alignment characters**
- excluded taxa: **23**
- final phylogeny: **2,194 genomes**
- IQ-TREE: **2.3.6**
- ModelFinder-selected model: **GTR+F+I+R10** by BIC (also selected by AIC/AICc)
- final optimized log-likelihood: **-10,658,044.645**
- support requested: **1,000 ultrafast bootstrap + 1,000 SH-aLRT replicates**
- random seed: **42**
- final production IQ-TREE log used **48 threads**
- ModelFinder estimated approximately **206,991 MB RAM** for the filtered alignment

The final production tree is:

```text
Figure1B_Peru_Cjejuni_core_IQTree_qc50.treefile
```

## Step 1 — initial annotation, pangenome and tree submission

```bash
./submit_peru2217_pirate_iqtree_v5.sh
```

The launcher:

1. reads the human and Peru source tables
2. creates an explicit set of 2,217 pubMLST IDs
3. maps each ID to exactly one assembly
4. submits batched Prokka annotation
5. verifies one GFF per expected isolate
6. runs PIRATE with DIAMOND
7. copies and validates the core alignment
8. submits IQ-TREE

The first IQ-TREE attempt should be treated as historical provenance rather than the final recommended resource configuration. The pangenome stage itself completed successfully and did not need repeating.

## Step 2 — taxon-level alignment QC

The initial PIRATE core alignment caused IQ-TREE to reject several extremely incomplete sequences. We therefore applied a transparent taxon-QC rule: **exclude a genome when >50% of its core-alignment characters are missing or ambiguous**.

Use:

```bash
./resubmit_cjejuni_iqtree_qc50_v2.sh
```

The QC step is deliberately streaming. Do not load the complete alignment into a Python list on the login node: the FASTA is several GB and an early non-streaming implementation was killed by the operating system.

The 23 excluded pubMLST IDs were:

```text
105612
108462
108485
108506
108514
108516
108520
108542
108544
151519
80361
80415
80638
80642
80643
80675
80684
80687
80700
80705
80710
80712
80718
```

The retained alignment contains **2,194 sequences**.

## Step 3 — final high-memory IQ-TREE run

A post-QC IQ-TREE run requesting 160 GB RAM OOM-killed during ModelFinder. This was a genuine memory failure, not a PIRATE or alignment failure.

The repository contains:

```bash
sbatch run_cjejuni_iqtree_qc50_240G.sh
```

as the minimum high-memory recovery template. The final completed production log used a larger-memory node and 48 threads; for exact replication on the same cluster, request enough RAM to exceed the ~207 GB ModelFinder estimate and use the available allocated CPU count. For example:

```bash
#SBATCH --cpus-per-task=48
#SBATCH --mem=240G
```

followed by:

```bash
iqtree2 \
  -s Peru_Cjejuni_PIRATE_core_alignment_qc50.fasta \
  -m MFP \
  -B 1000 \
  -alrt 1000 \
  -T 48 \
  -safe \
  -seed 42 \
  --prefix Peru_Cjejuni_PIRATE_core_IQTree_qc50
```

## IQ-TREE warnings retained in the audit trail

IQ-TREE reported that 146/2,194 sequences failed its composition chi-square test. These taxa were **not removed**, because this warning was not the pre-specified taxon-QC criterion and was not the cause of the earlier hard failure.

IQ-TREE also identified eight exact duplicate sequences. These were temporarily collapsed internally during optimization and reattached to the final tree. The biological/tree population remains **n=2,194**, not the number of unique sequence patterns used internally.

## Reproducibility principle

If IQ-TREE fails after PIRATE has produced a validated alignment, do not rerun Prokka/PIRATE unless the alignment itself is shown to be invalid. Treat annotation, pangenome construction, alignment QC and phylogenetic inference as separate recoverable stages.
