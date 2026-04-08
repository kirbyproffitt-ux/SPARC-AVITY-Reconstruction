# SPARC-AVITY-Reconstruction

Deterministic reconstruction of SPARC galaxy rotation curves using a fixed operator framework.

## Claim

This code reproduces SPARC galaxy rotation curves without per-galaxy tuning.

No per-galaxy parameter tuning is used.

Tier 1 galaxies achieve ~0.004 dex RMS residuals.

## Data Source

SPARC dataset (Lelli et al. 2016):
http://astroweb.cwru.edu/SPARC/

Download the dataset and place galaxy files in a local directory.

## How to Run

1. Place SPARC data in a folder (e.g., `data/`)
2. Run:
3. python reconstruct_sparc.py
4. python avity_plot_generator.py
