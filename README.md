# SPARC-AVITY-Reconstruction

Deterministic reconstruction of SPARC galaxy rotation curves using a fixed operator framework.

## Break My Code

This repository contains a deterministic reconstruction of SPARC galaxy rotation curve results using a fixed operator framework.

The goal is reproducibility.

If you find:

- implementation errors

- numerical inconsistencies

- validation failures

- galaxies that fail reconstruction

- discrepancies with the canonical reference tables

please open an Issue.

I am specifically interested in identifying failure cases and edge cases.

If you can identify a galaxy, parameter regime, reconstruction path, or numerical condition that breaks reproducibility, please document it and open an Issue.

Negative results are just as valuable as successful reproductions.

## Current Result

Using the archived reconstruction:

- No per-galaxy parameter tuning is used

- Tier 1 galaxies achieve approximately 0.004 dex RMS residuals

- Results are intended to reproduce the canonical reference tables

## Data Source

## Zenodo Archive

START HERE Dataset:

https://zenodo.org/records/19162721

DOI:

10.5281/zenodo.19162720

SPARC dataset (Lelli et al. 2016):
http://astroweb.cwru.edu/SPARC/

Download the dataset and place galaxy files in a local directory.

## How to Run

1. Place SPARC data in a folder (e.g., `data/`)
2. Run:
3. python reconstruct_sparc.py
4. python avity_plot_generator.py
