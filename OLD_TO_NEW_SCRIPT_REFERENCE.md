# Complete Old-to-New Script Reference

**Quick Lookup Table for Migrating from Old Codebase**

This document provides a complete mapping of all old scripts to their new refactored equivalents. Use Ctrl+F / Cmd+F to find your old script name.

---

## How to Use This Reference

1. **Find your old script** in the table below
2. **Look up the new script** in the "New Location" column
3. **Check the command** in the "Migration Command" column
4. **See the notes** for any special considerations

---

## Complete Script Mapping

### A

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `analyze_gcm_results.py` | `analysis/result_analyzer.py` | `python analysis/result_analyzer.py --data-type gcm` | GCM result analysis |
| `analyze_saved_solutions.py` | `analysis/result_analyzer.py` | `python analysis/result_analyzer.py --data-type solutions` | Solution analysis |

---

### C

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `create_combined_figure.py` | `visualization/manuscript_figures.py` | `python visualization/manuscript_figures.py --figure combined` | Manuscript figure generation |

---

### F

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `FASK_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method FASK` | Benchmark comparison |
| `FASK_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method FASK` | Time undersampling study |
| `fmri_experiment.py` | `real_data/fmri_experiment.py` | `python real_data/fmri_experiment.py` | Moved to real_data folder |

---

### G

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `GIMME_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method GIMME` | Benchmark comparison |
| `GIMME_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method GIMME` | Multiple RASL runs |
| `GIMME_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method GIMME` | Time undersampling study |

---

### L

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `lineal_stat_scan.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py` | Default variant (v1) |
| `lineal_stat_scan2.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --variant v2` | Alternative parameters |
| `lineal_stat_scan_config_test.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --mode config-test` | Config testing mode |
| `lineal_stat_scan_dataset.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --mode dataset` | Dataset scan mode |
| `linear_stat_continious_weights.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --continuous-weights` | Continuous weight mode |
| `linear_stat_continious_weights_n5.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --continuous-weights --nodes 5` | 5 nodes |
| `linear_stat_continious_weights_n6.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --continuous-weights --nodes 6` | 6 nodes |
| `linear_stat_continious_weights_n7.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --continuous-weights --nodes 7` | 7 nodes |
| `linear_stat_continious_weights_n8.py` | `simulation/linear_stat_scanner.py` | `python simulation/linear_stat_scanner.py --continuous-weights --nodes 8` | 8 nodes |

---

### M

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `macaque_analysis.py` | `real_data/macaque_analysis.py` | `python real_data/macaque_analysis.py` | Moved to real_data folder |
| `MVAR_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method MVAR` | Benchmark comparison |
| `MVAR_multi_individual_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method MVAR --mode individual` | Individual RASL mode |
| `MVAR_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method MVAR` | Multiple RASL runs |
| `MVAR_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method MVAR` | Time undersampling study |
| `MVGC_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method MVGC` | Benchmark comparison |
| `MVGC_multi_indiv_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method MVGC --mode individual` | Individual RASL mode |
| `MVGC_mult_comb_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method MVGC --mode combined` | Combined RASL mode |
| `MVGC_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `python benchmarks/multi_rasl_runner.py --method MVGC` | Multiple RASL runs |
| `MVGC_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method MVGC` | Time undersampling study |

---

### P

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `PCMCI_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method PCMCI` | Benchmark comparison |
| `plot_fmri_enhanced.py` | `visualization/network_plots.py` | `python visualization/network_plots.py --data-source fmri` | fMRI network plots |
| `plot_gcm_enhanced.py` | `visualization/network_plots.py` | `python visualization/network_plots.py --data-source gcm` | GCM network plots |
| `plot_manuscript_compact.py` | `visualization/network_plots.py` | `python visualization/network_plots.py --data-source rasl` | Compact manuscript plots |

---

### Q

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `quick_analyze_results.py` | `analysis/result_analyzer.py` | `python analysis/result_analyzer.py --quick` | Quick analysis mode |

---

### R

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `RASL_fig4.py` | `benchmarks/benchmark_runner.py` | `python benchmarks/benchmark_runner.py --method RASL` | Benchmark comparison |
| `Read_simulation_res.py` | `analysis/result_analyzer.py` | `python analysis/result_analyzer.py --data-type simulation` | Simulation result reading |
| `Read_simulation_res_optN.py` | `analysis/result_analyzer.py` | `python analysis/result_analyzer.py --data-type simulation --optimize-n` | Optimized N variant |
| `roebroeck_gcm.py` | `real_data/roebroeck_gcm.py` | `python real_data/roebroeck_gcm.py` | Moved to real_data folder |

---

### S

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `slurm_FASK_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method FASK --slurm` | SLURM mode enabled |
| `slurm_GIMME_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method GIMME --slurm` | SLURM mode enabled |
| `slurm_MVAR_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method MVAR --slurm` | SLURM mode enabled |
| `slurm_MVGC_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `python benchmarks/time_undersampling.py --method MVGC --slurm` | SLURM mode enabled |

---

### V

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `VAR_analyze_delta.py` | `experiments/var_analyzer.py` | `python experiments/var_analyzer.py --analysis-type delta` | Delta analysis |
| `VAR_analyze_hyperparameters.py` | `experiments/var_analyzer.py` | `python experiments/var_analyzer.py --analysis-type hyperparameters` | Hyperparameter analysis |
| `VAR_analyze_orientation_only.py` | `experiments/var_analyzer.py` | `python experiments/var_analyzer.py --analysis-type orientation` | Orientation analysis |

---

### W

| Old Script | New Location | Migration Command | Notes |
|------------|--------------|-------------------|-------|
| `weighted.py` | `simulation/weighted_solver.py` | `python simulation/weighted_solver.py` | Base weighted solver |
| `weighted_optN.py` | `simulation/weighted_solver.py` | `python simulation/weighted_solver.py --optimize-density` | Density optimization |
| `weighted_then_drasl.py` | `simulation/weighted_solver.py` | `python simulation/weighted_solver.py --method weighted-drasl` | Combined approach |

---

## Scripts Moved to Legacy (Archived)

These scripts have been moved to `scripts/legacy/` and are no longer actively maintained. They are preserved for reference and reproducibility of old results.

### Experimental Variants

| Old Script | Legacy Location | Reason |
|------------|----------------|--------|
| `*.retry` scripts | `legacy/experimental/` | Failed experiment retries |
| `*_v2.py`, `*_v3.py` scripts | `legacy/variants/` | Experimental parameter variants |
| `*_test.py` scripts | `legacy/testing/` | Old test scripts |

### Old Directory Contents

| Old Script | Legacy Location | Reason |
|------------|----------------|--------|
| `old/*.py` | `legacy/archive_old/` | Pre-refactoring legacy code |

---

## Detailed Migration Examples

### Example 1: Fig4 Benchmark Script

**Old way**:

```bash
# Had to run each method separately
python FASK_fig4.py --nodes 10 --samples 500 --output results/fask.png
python GIMME_fig4.py --nodes 10 --samples 500 --output results/gimme.png
python MVAR_fig4.py --nodes 10 --samples 500 --output results/mvar.png
python MVGC_fig4.py --nodes 10 --samples 500 --output results/mvgc.png
python PCMCI_fig4.py --nodes 10 --samples 500 --output results/pcmci.png
python RASL_fig4.py --nodes 10 --samples 500 --output results/rasl.png
```

**New way**:

```bash
# Single script runs all methods
python benchmarks/benchmark_runner.py \
    --method all \
    --nodes 10 \
    --samples 500 \
    --output-dir results/
```

### Example 2: Enhanced Plotting

**Old way**:

```bash
# Different scripts for different data sources
python plot_fmri_enhanced.py --input data/fmri.pkl --output fig_fmri.pdf
python plot_gcm_enhanced.py --input data/gcm.pkl --output fig_gcm.pdf
python plot_manuscript_compact.py --input data/rasl.pkl --output fig_rasl.pdf
```

**New way**:

```bash
# Single script with data source parameter
python visualization/network_plots.py \
    --data-source fmri \
    --input data/fmri.pkl \
    --output fig_fmri.pdf

python visualization/network_plots.py \
    --data-source gcm \
    --input data/gcm.pkl \
    --output fig_gcm.pdf

python visualization/network_plots.py \
    --data-source rasl \
    --input data/rasl.pkl \
    --output fig_rasl.pdf

# Or generate all at once
python visualization/network_plots.py \
    --data-source all \
    --input-dir data/ \
    --output-dir figures/
```

### Example 3: Time Undersampling with SLURM

**Old way**:

```bash
# Local version
python MVAR_time_undersampling_data.py --undersampling 2,4,8

# Separate SLURM version
python slurm_MVAR_time_undersampling_data.py --undersampling 2,4,8
```

**New way**:

```bash
# Same script, just add --slurm flag
python benchmarks/time_undersampling.py \
    --method MVAR \
    --undersampling 2,4,8

# For SLURM
python benchmarks/time_undersampling.py \
    --method MVAR \
    --undersampling 2,4,8 \
    --slurm \
    --submit
```

### Example 4: Linear Statistics Scanning

**Old way**:

```bash
# Different scripts for different node counts
python linear_stat_continious_weights_n5.py --samples 1000
python linear_stat_continious_weights_n6.py --samples 1000
python linear_stat_continious_weights_n7.py --samples 1000
python linear_stat_continious_weights_n8.py --samples 1000
```

**New way**:

```bash
# Single script with node parameter
python simulation/linear_stat_scanner.py \
    --continuous-weights \
    --nodes 5,6,7,8 \
    --samples 1000
```

### Example 5: VAR Analysis

**Old way**:

```bash
# Three separate analysis scripts
python VAR_analyze_delta.py --input results/var_data.csv
python VAR_analyze_hyperparameters.py --input results/var_data.csv
python VAR_analyze_orientation_only.py --input results/var_data.csv
```

**New way**:

```bash
# Single script with analysis type
python experiments/var_analyzer.py \
    --analysis-type delta \
    --input results/var_data.csv

python experiments/var_analyzer.py \
    --analysis-type hyperparameters \
    --input results/var_data.csv

python experiments/var_analyzer.py \
    --analysis-type orientation \
    --input results/var_data.csv

# Or run all analyses at once
python experiments/var_analyzer.py \
    --analysis-type all \
    --input results/var_data.csv \
    --output-dir analysis/
```

---

## Parameter Name Changes

Some parameter names have been standardized across scripts:

| Old Parameter | New Parameter | Scripts Affected |
|---------------|---------------|------------------|
| `--output_dir` | `--output-dir` | All scripts (underscore → hyphen) |
| `--input_file` | `--input` | All scripts (shortened) |
| `--num_nodes` | `--nodes` | Simulation scripts |
| `--num_samples` | `--samples` | Simulation scripts |
| `--PreFix` | `--method` | Benchmark scripts |
| `--GT_density` | `--density` | Weighted solver |

**Note**: Most scripts accept both old and new parameter names for backward compatibility, but new names are recommended.

---

## Import Path Changes

If you're importing functions from these scripts in custom code:

### Old imports:

```python
from gunfolds.scripts.plot_fmri_enhanced import plot_network_circular
from gunfolds.scripts.MVAR_fig4 import create_boxplot
from gunfolds.scripts.VAR_analyze_delta import compute_delta_metrics
```

### New imports:

```python
from gunfolds.scripts.visualization.network_plots import plot_network_circular
from gunfolds.scripts.benchmarks.benchmark_runner import create_boxplot
from gunfolds.scripts.experiments.var_analyzer import compute_delta_metrics
```

---

## Finding Archived Scripts

If you need to access old scripts (for reproducing old results exactly):

```bash
# Old scripts are in scripts/legacy/
cd gunfolds/scripts/legacy/

# Structure:
legacy/
├── archive_old/          # Contents of old/ directory
├── experimental/         # Failed experiments (*.retry)
├── testing/             # Old test_*.py scripts
└── variants/            # Numbered variants (*2.py, *3.py)

# To use an old script:
python legacy/experimental/MVAR_fig4_variant3.py

# Or copy to current directory
cp legacy/experimental/MVAR_fig4_variant3.py ./
python MVAR_fig4_variant3.py
```

---

## Getting Help

### For specific scripts:

All new scripts have comprehensive `--help`:

```bash
python benchmarks/benchmark_runner.py --help
python visualization/network_plots.py --help
python experiments/var_analyzer.py --help
```

### Documentation:

- **Migration Guide**: `MIGRATION.md` (comprehensive migration instructions)
- **Quick Start**: `docs/QUICKSTART.md` (getting started with new structure)
- **Hyperparameter Tuning**: `docs/experiments/hyperparameter_tuning.md`
- **Cluster Guide**: `docs/experiments/cluster_guide.md`

### Support:

- **GitHub Issues**: Report bugs or missing functionality
- **GitHub Discussions**: Ask questions about migration

---

## Quick Reference Summary

### Most Common Migrations

```bash
# Benchmarks (all fig4 scripts)
OLD: python [METHOD]_fig4.py
NEW: python benchmarks/benchmark_runner.py --method [METHOD]

# Enhanced plotting
OLD: python plot_[TYPE]_enhanced.py
NEW: python visualization/network_plots.py --data-source [TYPE]

# Time undersampling
OLD: python [METHOD]_time_undersampling_data.py
NEW: python benchmarks/time_undersampling.py --method [METHOD]

# SLURM time undersampling
OLD: python slurm_[METHOD]_time_undersampling_data.py
NEW: python benchmarks/time_undersampling.py --method [METHOD] --slurm

# Linear stat scanning
OLD: python linear_stat_continious_weights_n[N].py
NEW: python simulation/linear_stat_scanner.py --continuous-weights --nodes [N]

# VAR analysis
OLD: python VAR_analyze_[TYPE].py
NEW: python experiments/var_analyzer.py --analysis-type [TYPE]

# Result analysis
OLD: python analyze_[TYPE]_results.py
NEW: python analysis/result_analyzer.py --data-type [TYPE]

# Weighted solver
OLD: python weighted_optN.py
NEW: python simulation/weighted_solver.py --optimize-density
```

---

## Verification

To verify your migration is correct:

1. **Run both old and new scripts** on same input
2. **Compare outputs**: Results should be identical (or very close)
3. **Check file sizes**: Output files should be similar size
4. **Validate metrics**: Precision, recall, F1 should match

Example verification:

```bash
# Run old script (from legacy)
python legacy/experimental/MVAR_fig4.py \
    --nodes 10 \
    --samples 100 \
    --output results_old/

# Run new script
python benchmarks/benchmark_runner.py \
    --method MVAR \
    --nodes 10 \
    --samples 100 \
    --output-dir results_new/

# Compare results
python utils/compare_results.py \
    --old results_old/ \
    --new results_new/ \
    --tolerance 1e-6
```

---

**Questions?** See `MIGRATION.md` for detailed migration instructions or open a GitHub issue.

**Version**: 2.0.0  
**Last Updated**: December 2, 2025  
**Maintainer**: Gunfolds Development Team

