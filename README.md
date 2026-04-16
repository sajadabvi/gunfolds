# gunfolds

A Python toolkit for causal structure learning from undersampled time series,
with a focus on scalable constraint-based methods for real-world graph sizes.

## My Contributions

This repository is developed and maintained primarily by me (Sajad Abavisani)
as part of my PhD research at Georgia Tech, in collaboration with the
[TReNDS / neuroneural lab](https://github.com/neuroneural).

Key contributions I built:

- **GRACE-C** — Reformulated rate-agnostic causal structure learning as an
  Answer Set Programming (ASP) constraint satisfaction problem, achieving a
  **1,000× speedup** over prior methods (17 hrs → 6 sec on 6-node graphs).
  Scales to 100+ node graphs via SCC decomposition. Presented as an
  [Oral at ICLR 2023](https://openreview.net/forum?id=B_pCIsX8KL_) (top 25%).

- **RnR** — A meta-solver that refines the output of *any* causal discovery
  algorithm by modeling undersampling effects via ASP. Improves F1 by 45%
  over SOTA baselines (PCMCI, FASK, MVGC, GIMME) on real fMRI data.
  Under review at ICML 2026.

- **ION-C** — Causal learning from overlapping datasets with non-co-measured
  variables. Proved soundness and completeness; scales from 4–6 nodes
  (prior limit) to 25+ nodes.

- **dRASL** — Demonstrated that deliberate undersampling can reduce causal
  uncertainty by up to 4 orders of magnitude. Published at CLeaR/PMLR 2023.

## Installation

```bash
pip install gunfolds
```

For optional dependencies (`graph-tool`, `PyGObject`) see below.

### graph-tool

```bash
# conda
conda install -c conda-forge graph-tool

# brew (macOS)
brew install graph-tool
```

### PyGObject (only needed for the `gtool` module)

```bash
# macOS
brew install pygobject3 gtk+3

# Other platforms
# https://pygobject.readthedocs.io/en/latest/getting_started.html
```

## Documentation

Full API reference and guides: [neuroneural.github.io/gunfolds](https://neuroneural.github.io/gunfolds/)

## Publications

- Abavisani et al. **GRACE-C: Generalized Rate Agnostic Causal Estimation via
  Constraints.** ICLR 2023 (Oral, Top 25%)
- Abavisani et al. **RnR: A Meta-Solver for Causal Discovery in Undersampled
  Time Series.** Under review, ICML 2026
- Solovyeva, Danks, Abavisani, Plis. **Causal Learning through Deliberate
  Undersampling.** CLeaR / PMLR 2023
- Nair et al. (incl. Abavisani). **ION-C: Integration of Overlapping Networks
  via Constraints.** arXiv 2024

## Acknowledgments

Supported by NSF IIS-1318759 and NIH 1R01MH129047.