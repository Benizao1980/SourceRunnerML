#!/usr/bin/env python3
"""Create small, fully synthetic cgMLST-like tables for the SourceRunnerML tutorial.

NOT real genomes, epidemiological evidence, or a scientific benchmark.
Requires only the Python standard library.
"""
import argparse
import csv
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="toy_data", help="Output directory")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    loci = [f"CAMP{i:04d}" for i in range(1, 121)]
    classes = ["Poultry", "Ruminant", "Pig"]

    def alleles(class_index):
        values = []
        for index in range(len(loci)):
            # Artificial class signal at 24 loci; remaining loci are noisy.
            # Allele identifiers are nominal labels, not genetic distances.
            if index < 24:
                allele = 100 + class_index * 100 + rng.randint(0, 6)
            else:
                allele = 10 + rng.randint(0, 75)
            values.append("" if rng.random() < 0.04 else str(allele))
        return values

    train = outdir / "train.tsv"
    with train.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "source"] + loci)
        for index, source in enumerate(classes):
            for sample in range(60):
                writer.writerow([f"REF_{index}_{sample:03d}", source] + alleles(index))

    predict = outdir / "predict.tsv"
    with predict.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["id", "demo_truth"] + loci)
        for index, source in enumerate(classes):
            for sample in range(10):
                writer.writerow([f"QUERY_{index}_{sample:03d}", source] + alleles(index))

    print(f"Synthetic training rows: 180; prediction rows: 30; loci: {len(loci)}")
    print(f"Training: {train}")
    print(f"Prediction: {predict}")
    print("These are simulated toy profiles. Do not publish performance from them.")


if __name__ == "__main__":
    main()
