# Gunfolds Codebase Migration Guide

**Version**: 2.0.0 Refactoring  
**Date**: December 2025  
**Status**: Complete Reorganization

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Reference: Old → New Script Mapping](#quick-reference-old--new-script-mapping)
3. [Detailed Migration by Category](#detailed-migration-by-category)
4. [Command-Line Examples](#command-line-examples)
5. [Breaking Changes](#breaking-changes)
6. [New Directory Structure](#new-directory-structure)
7. [Frequently Asked Questions](#frequently-asked-questions)

---

## Overview

### What Changed?

The `gunfolds/scripts/` directory has been completely reorganized to eliminate redundancy and improve maintainability. **118+ scripts** have been consolidated into **~33 core modules** organized by function.

---

## Quick Reference: Old → New Script Mapping

### 📊 Visualization & Plotting

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `plot_fmri_enhanced.py` | `visualization/network_plots.py` | `--data-source fmri` |
| `plot_gcm_enhanced.py` | `visualization/network_plots.py` | `--data-source gcm` |
| `plot_manuscript_compact.py` | `visualization/network_plots.py` | `--data-source rasl` |
| `create_combined_figure.py` | `visualization/manuscript_figures.py` | `--figure combined` |

### 📈 Benchmark & Method Comparison Scripts

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `FASK_fig4.py` | `benchmarks/benchmark_runner.py` | `--method FASK` |
| `GIMME_fig4.py` | `benchmarks/benchmark_runner.py` | `--method GIMME` |
| `MVAR_fig4.py` | `benchmarks/benchmark_runner.py` | `--method MVAR` |
| `MVGC_fig4.py` | `benchmarks/benchmark_runner.py` | `--method MVGC` |
| `PCMCI_fig4.py` | `benchmarks/benchmark_runner.py` | `--method PCMCI` |
| `RASL_fig4.py` | `benchmarks/benchmark_runner.py` | `--method RASL` |

### ⏱️ Time Undersampling Experiments

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `FASK_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method FASK` |
| `GIMME_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method GIMME` |
| `MVAR_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method MVAR` |
| `MVGC_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method MVGC` |
| `slurm_FASK_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method FASK --slurm` |
| `slurm_GIMME_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method GIMME --slurm` |
| `slurm_MVAR_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method MVAR --slurm` |
| `slurm_MVGC_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | `--method MVGC --slurm` |

### 🔄 Multiple RASL Experiments

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `GIMME_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method GIMME` |
| `MVAR_multi_individual_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method MVAR --mode individual` |
| `MVAR_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method MVAR` |
| `MVGC_multi_indiv_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method MVGC --mode individual` |
| `MVGC_mult_comb_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method MVGC --mode combined` |
| `MVGC_multiple_rasl.py` | `benchmarks/multi_rasl_runner.py` | `--method MVGC` |

### 📊 Linear Statistics & Scanning

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `lineal_stat_scan.py` | `simulation/linear_stat_scanner.py` | Default settings |
| `lineal_stat_scan2.py` | `simulation/linear_stat_scanner.py` | `--variant v2` |
| `lineal_stat_scan_config_test.py` | `simulation/linear_stat_scanner.py` | `--mode config-test` |
| `lineal_stat_scan_dataset.py` | `simulation/linear_stat_scanner.py` | `--mode dataset` |
| `linear_stat_continious_weights*.py` (5 variants) | `simulation/linear_stat_scanner.py` | `--continuous-weights --nodes N` |

### ⚖️ Weighted Solver Scripts

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `weighted.py` | `simulation/weighted_solver.py` | Default mode |
| `weighted_optN.py` | `simulation/weighted_solver.py` | `--optimize-density` |
| `weighted_then_drasl.py` | `simulation/weighted_solver.py` | `--method weighted-drasl` |

### 🔬 VAR Analysis Scripts

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `VAR_analyze_delta.py` | `experiments/var_analyzer.py` | `--analysis-type delta` |
| `VAR_analyze_hyperparameters.py` | `experiments/var_analyzer.py` | `--analysis-type hyperparameters` |
| `VAR_analyze_orientation_only.py` | `experiments/var_analyzer.py` | `--analysis-type orientation` |

### 📋 Result Analysis Scripts

| Old Script | New Script | Usage |
|------------|-----------|-------|
| `analyze_gcm_results.py` | `analysis/result_analyzer.py` | `--data-type gcm` |
| `analyze_saved_solutions.py` | `analysis/result_analyzer.py` | `--data-type solutions` |
| `quick_analyze_results.py` | `analysis/result_analyzer.py` | `--quick` |
| `Read_simulation_res.py` | `analysis/result_analyzer.py` | `--data-type simulation` |
| `Read_simulation_res_optN.py` | `analysis/result_analyzer.py` | `--data-type simulation --optimize-n` |

### 🧪 Real Data Experiments (No Change - Moved)

| Old Script | New Location | Notes |
|------------|-------------|-------|
| `fmri_experiment.py` | `real_data/fmri_experiment.py` | Moved to new folder |
| `roebroeck_gcm.py` | `real_data/roebroeck_gcm.py` | Moved to new folder |
| `macaque_analysis.py` | `real_data/macaque_analysis.py` | Moved to new folder |

### 💻 Cluster Scripts (Reorganized)

| Old Script | New Location | Notes |
|------------|-------------|-------|
| `slurm_*.py` scripts | `cluster/` directory | All SLURM scripts consolidated |
| Hardcoded job scripts | `cluster/job_submission.sh` | Template-based submission |

### 🗂️ Archived Scripts

| Old Script | New Location | Reason |
|------------|-------------|--------|
| Scripts in `old/` | `legacy/archive_old/` | Pre-existing legacy code |
| `test_*.py` scripts | `legacy/testing/` | Old test scripts |
| `*.retry` scripts | `legacy/experimental/` | Failed experiment variants |
| Numbered variants (`*2.py`, `*3.py`) | `legacy/variants/` | Old parameter experiments |

---

## Detailed Migration by Category

### 1. Enhanced Plotting Scripts → `visualization/network_plots.py`

#### Old Workflow

```python
# Had to use different scripts for different data sources
python gunfolds/scripts/plot_fmri_enhanced.py
python gunfolds/scripts/plot_gcm_enhanced.py  
python gunfolds/scripts/plot_manuscript_compact.py
```

#### New Workflow

```python
# Single unified script with data source parameter
python gunfolds/scripts/visualization/network_plots.py --data-source fmri
python gunfolds/scripts/visualization/network_plots.py --data-source gcm
python gunfolds/scripts/visualization/network_plots.py --data-source rasl
```

#### Key Changes

- All three scripts shared 99% identical code (530/412/230 lines)
- Functions `plot_network_circular()`, `plot_heatmap_enhanced()`, `plot_top_connections()` now in single module
- Data loading logic parameterized by `--data-source` flag

#### Migration Example

```python
# OLD: plot_fmri_enhanced.py
python plot_fmri_enhanced.py --output results/fmri_plot.png

# NEW: Equivalent command
python visualization/network_plots.py \
    --data-source fmri \
    --output results/fmri_plot.png
```

---

### 2. Fig4 Benchmark Scripts → `benchmarks/benchmark_runner.py`

#### Old Workflow

```bash
# Had to run 6 separate scripts for 6 methods
python FASK_fig4.py
python GIMME_fig4.py
python MVAR_fig4.py
python MVGC_fig4.py
python PCMCI_fig4.py
python RASL_fig4.py
```

#### New Workflow

```bash
# Single script with method parameter
python benchmarks/benchmark_runner.py --method FASK
python benchmarks/benchmark_runner.py --method GIMME
python benchmarks/benchmark_runner.py --method MVAR
python benchmarks/benchmark_runner.py --method MVGC
python benchmarks/benchmark_runner.py --method PCMCI
python benchmarks/benchmark_runner.py --method RASL

# Or run all methods at once
python benchmarks/benchmark_runner.py --method all
```

#### Key Changes

- All 6 scripts had identical boxplot logic (~350 lines each)
- Only differed in `PreFix` variable (method name) and data file paths
- Shared plotting code extracted to common module

#### Migration Example

```python
# OLD: MVAR_fig4.py with custom parameters
python MVAR_fig4.py --nodes 10 --samples 100

# NEW: Equivalent command
python benchmarks/benchmark_runner.py \
    --method MVAR \
    --nodes 10 \
    --samples 100
```

---

### 3. Time Undersampling → `benchmarks/time_undersampling.py`

#### Old Workflow

```bash
# Local execution: 4 separate scripts
python FASK_time_undersampling_data.py
python GIMME_time_undersampling_data.py
python MVAR_time_undersampling_data.py
python MVGC_time_undersampling_data.py

# SLURM execution: 4 additional scripts
python slurm_FASK_time_undersampling_data.py
python slurm_GIMME_time_undersampling_data.py
python slurm_MVAR_time_undersampling_data.py
python slurm_MVGC_time_undersampling_data.py
```

#### New Workflow

```bash
# Local execution
python benchmarks/time_undersampling.py --method FASK
python benchmarks/time_undersampling.py --method GIMME

# SLURM execution (same script!)
python benchmarks/time_undersampling.py --method FASK --slurm
python benchmarks/time_undersampling.py --method GIMME --slurm

# Batch all methods
python benchmarks/time_undersampling.py --method all --slurm
```

#### Key Changes

- 8 scripts → 1 unified script
- SLURM vs local execution controlled by `--slurm` flag
- Automatic job array creation for batch execution

#### Migration Example

```python
# OLD: slurm_MVAR_time_undersampling_data.py
python slurm_MVAR_time_undersampling_data.py \
    --undersampling 2,4,8 \
    --nodes 5

# NEW: Equivalent command
python benchmarks/time_undersampling.py \
    --method MVAR \
    --slurm \
    --undersampling 2,4,8 \
    --nodes 5
```

---

### 4. Linear Stat Scanning → `simulation/linear_stat_scanner.py`

#### Old Workflow

```bash
# 9 different scripts for different variants
python lineal_stat_scan.py
python lineal_stat_scan2.py  # Different defaults
python lineal_stat_scan_config_test.py
python lineal_stat_scan_dataset.py
python linear_stat_continious_weights.py
python linear_stat_continious_weights_n5.py
python linear_stat_continious_weights_n6.py
python linear_stat_continious_weights_n7.py
python linear_stat_continious_weights_n8.py
```

#### New Workflow

```bash
# Single script with parameters
python simulation/linear_stat_scanner.py  # Default (v1)
python simulation/linear_stat_scanner.py --variant v2  # Old scan2.py
python simulation/linear_stat_scanner.py --mode config-test
python simulation/linear_stat_scanner.py --mode dataset

# Continuous weights with different node counts
python simulation/linear_stat_scanner.py --continuous-weights --nodes 5
python simulation/linear_stat_scanner.py --continuous-weights --nodes 6
python simulation/linear_stat_scanner.py --continuous-weights --nodes 7
python simulation/linear_stat_scanner.py --continuous-weights --nodes 8
```

#### Key Changes

- 9 scripts → 1 parameterized script
- Node count now a command-line argument
- Scan modes unified under `--mode` flag

#### Migration Example

```python
# OLD: linear_stat_continious_weights_n7.py (hardcoded 7 nodes)
python linear_stat_continious_weights_n7.py --samples 1000

# NEW: Equivalent command
python simulation/linear_stat_scanner.py \
    --continuous-weights \
    --nodes 7 \
    --samples 1000
```

---

### 5. VAR Analysis → `experiments/var_analyzer.py`

#### Old Workflow

```bash
# 3 separate scripts for different analysis types
python VAR_analyze_delta.py
python VAR_analyze_hyperparameters.py
python VAR_analyze_orientation_only.py
```

#### New Workflow

```bash
# Single script with analysis type parameter
python experiments/var_analyzer.py --analysis-type delta
python experiments/var_analyzer.py --analysis-type hyperparameters
python experiments/var_analyzer.py --analysis-type orientation

# Run all analyses
python experiments/var_analyzer.py --analysis-type all
```

#### Key Changes

- Shared matplotlib plotting code extracted
- Consistent output format across analysis types
- Unified metric collection

#### Migration Example

```python
# OLD: VAR_analyze_hyperparameters.py
python VAR_analyze_hyperparameters.py \
    --input results/var_data.csv \
    --output plots/hyperparams.png

# NEW: Equivalent command
python experiments/var_analyzer.py \
    --analysis-type hyperparameters \
    --input results/var_data.csv \
    --output plots/hyperparams.png
```

---

### 6. Result Analysis → `analysis/result_analyzer.py`

#### Old Workflow

```bash
# 5 different analysis scripts
python analyze_gcm_results.py
python analyze_saved_solutions.py
python quick_analyze_results.py  # Just a wrapper
python Read_simulation_res.py
python Read_simulation_res_optN.py
```

#### New Workflow

```bash
# Single unified analyzer
python analysis/result_analyzer.py --data-type gcm
python analysis/result_analyzer.py --data-type solutions
python analysis/result_analyzer.py --data-type simulation
python analysis/result_analyzer.py --data-type simulation --optimize-n

# Quick mode (minimal output)
python analysis/result_analyzer.py --data-type gcm --quick
```

#### Key Changes

- Unified interface for all result types
- Automatic format detection
- Consistent metrics across data types

#### Migration Example

```python
# OLD: analyze_gcm_results.py
python analyze_gcm_results.py \
    --results-dir results/gcm/ \
    --output summary.csv

# NEW: Equivalent command
python analysis/result_analyzer.py \
    --data-type gcm \
    --results-dir results/gcm/ \
    --output summary.csv
```

---

### 7. Weighted Solver → `simulation/weighted_solver.py`

#### Old Workflow

```bash
# 3 separate scripts
python weighted.py  # Base version
python weighted_optN.py  # With density optimization
python weighted_then_drasl.py  # Combined approach
```

#### New Workflow

```bash
# Single script with modes
python simulation/weighted_solver.py  # Base mode
python simulation/weighted_solver.py --optimize-density
python simulation/weighted_solver.py --method weighted-drasl
```

#### Key Changes

- GT_density parameter now optional flag
- Combined weighted+drasl approach integrated
- Consistent output format

#### Migration Example

```python
# OLD: weighted_optN.py
python weighted_optN.py --nodes 10 --density 0.3

# NEW: Equivalent command
python simulation/weighted_solver.py \
    --nodes 10 \
    --optimize-density \
    --density 0.3
```

---

### 8. Multiple RASL → `benchmarks/multi_rasl_runner.py`

#### Old Workflow

```bash
# 6 different RASL scripts
python GIMME_multiple_rasl.py
python MVAR_multi_individual_rasl.py
python MVAR_multiple_rasl.py
python MVGC_multi_indiv_rasl.py
python MVGC_mult_comb_rasl.py
python MVGC_multiple_rasl.py
```

#### New Workflow

```bash
# Unified runner
python benchmarks/multi_rasl_runner.py --method GIMME
python benchmarks/multi_rasl_runner.py --method MVAR --mode individual
python benchmarks/multi_rasl_runner.py --method MVAR --mode multiple
python benchmarks/multi_rasl_runner.py --method MVGC --mode individual
python benchmarks/multi_rasl_runner.py --method MVGC --mode combined
```

#### Key Changes

- Method and mode as parameters
- Consistent RASL execution logic
- Unified result collection

---

## Command-Line Examples

### Example 1: Running Benchmarks for All Methods

**Old way** (run 6 scripts manually):

```bash
python FASK_fig4.py --output results/fask.png
python GIMME_fig4.py --output results/gimme.png
python MVAR_fig4.py --output results/mvar.png
python MVGC_fig4.py --output results/mvgc.png
python PCMCI_fig4.py --output results/pcmci.png
python RASL_fig4.py --output results/rasl.png
```

**New way** (single batch command):

```bash
python benchmarks/benchmark_runner.py \
    --method all \
    --output-dir results/ \
    --format png
```

---

### Example 2: Time Undersampling Study on Cluster

**Old way** (2 separate scripts):

```bash
# Generate SLURM scripts
python slurm_MVAR_time_undersampling_data.py --generate-only

# Submit manually
sbatch slurm_mvar_undersampling.sh
```

**New way** (integrated):

```bash
# Generate and submit in one command
python benchmarks/time_undersampling.py \
    --method MVAR \
    --slurm \
    --submit \
    --undersampling 2,4,8,16 \
    --nodes 5,10,15
```

---

### Example 3: Creating Publication Figures

**Old way** (multiple scripts):

```bash
python plot_fmri_enhanced.py --output fig_fmri.pdf
python plot_gcm_enhanced.py --output fig_gcm.pdf
python create_combined_figure.py --inputs fig_fmri.pdf,fig_gcm.pdf
```

**New way** (streamlined):

```bash
# Generate all plots
python visualization/network_plots.py --data-source all --output-dir figures/

# Create combined manuscript figure
python visualization/manuscript_figures.py \
    --figure combined \
    --input-dir figures/ \
    --output manuscript_figure.pdf
```

---

### Example 4: VAR Hyperparameter Tuning Analysis

**Old way** (3 separate analyses):

```bash
python VAR_analyze_delta.py --input results/var_delta.csv
python VAR_analyze_hyperparameters.py --input results/var_hp.csv
python VAR_analyze_orientation_only.py --input results/var_orient.csv
```

**New way** (comprehensive analysis):

```bash
python experiments/var_analyzer.py \
    --analysis-type all \
    --input results/var_data.csv \
    --output-dir analysis/
```

---

## Breaking Changes

### 1. Import Paths

**Before:**

```python
from gunfolds.scripts.plot_fmri_enhanced import plot_network_circular
from gunfolds.scripts.MVAR_fig4 import create_boxplot
```

**After:**

```python
from gunfolds.scripts.visualization.network_plots import plot_network_circular
from gunfolds.scripts.benchmarks.benchmark_runner import create_boxplot
```

### 2. Function Signatures

Some functions have been standardized:

**Before:**

```python
plot_network_circular(data, title="FMRI Network")  # fmri version
plot_network_circular(gcm_data, graph_title="GCM")  # gcm version
```

**After (unified):**

```python
plot_network_circular(data, title="Network", data_source="fmri")
```

### 3. Output File Naming

Old scripts used inconsistent naming:

```bash
# Old outputs
MVAR_fig4_output.png
mvar_figure4_result.png
MVAR-Fig4-Final.png
```

New scripts use consistent naming:

```bash
# New outputs (predictable)
benchmark_MVAR_fig4.png
time_undersampling_MVAR_u2.png
```

### 4. Configuration Files

**Before:** Each script had hardcoded parameters

**After:** Unified config system

```bash
# Generate default config
python simulation/linear_stat_scanner.py --generate-config > config.yaml

# Use config
python simulation/linear_stat_scanner.py --config config.yaml
```

---

## New Directory Structure

```
gunfolds/
├── scripts/
│   ├── analysis/              # ← Result analysis tools
│   │   ├── result_analyzer.py
│   │   └── solution_comparer.py
│   ├── benchmarks/            # ← Method comparison experiments
│   │   ├── benchmark_runner.py
│   │   ├── time_undersampling.py
│   │   └── multi_rasl_runner.py
│   ├── experiments/           # ← VAR experiments & tuning
│   │   ├── var_hyperparameter_tuning.py
│   │   ├── var_analyzer.py
│   │   ├── var_collector.py
│   │   └── var_ruben_experiments.py
│   ├── visualization/         # ← All plotting functions
│   │   ├── network_plots.py
│   │   ├── manuscript_figures.py
│   │   └── plotting_utils.py
│   ├── simulation/            # ← Simulation & data generation
│   │   ├── linear_stat_scanner.py
│   │   ├── weighted_solver.py
│   │   └── data_generator.py
│   ├── real_data/             # ← Real dataset experiments
│   │   ├── fmri_experiment.py
│   │   ├── roebroeck_gcm.py
│   │   └── macaque_analysis.py
│   ├── cluster/               # ← SLURM/cluster scripts
│   │   ├── slurm_templates.py
│   │   └── job_submission.sh
│   ├── legacy/                # ← Archived old/experimental code
│   │   ├── archive_old/       # Old scripts from old/ directory
│   │   ├── experimental/      # Failed experiments (*.retry)
│   │   ├── testing/           # Old test_*.py scripts
│   │   └── variants/          # Numbered variants (*2.py, *3.py)
│   └── utils/
│       └── common_functions.py
└── docs/                      # ← Consolidated documentation
    ├── README.md
    ├── QUICKSTART.md
    └── experiments/
        ├── hyperparameter_tuning.md
        └── cluster_guide.md
```

### Where to Find Things

| What You Need | Where It Lives |
|---------------|----------------|
| Run benchmarks | `scripts/benchmarks/` |
| Analyze results | `scripts/analysis/` |
| Create plots | `scripts/visualization/` |
| VAR experiments | `scripts/experiments/` |
| Generate data | `scripts/simulation/` |
| Real data analysis | `scripts/real_data/` |
| Cluster jobs | `scripts/cluster/` |
| Old scripts | `scripts/legacy/` |
| Documentation | `docs/` |

---

## Frequently Asked Questions

### Q1: Where did my old script go?

**A:** Check the [Quick Reference table](#quick-reference-old--new-script-mapping) above. Most scripts are now part of unified modules with parameters. If not listed, it's in `scripts/legacy/`.

---

### Q2: Can I still use the old scripts?

**A:** Yes, temporarily. Deprecated wrapper scripts are provided that call the new unified scripts with a warning:

```bash
$ python plot_fmri_enhanced.py
⚠️  WARNING: This script is deprecated and will be removed in v2.1.0
    Please use: python visualization/network_plots.py --data-source fmri
```

These wrappers will be removed in the next major release.

---

### Q3: How do I find the right parameters for the new scripts?

**A:** All new scripts have comprehensive `--help`:

```bash
python benchmarks/benchmark_runner.py --help
python visualization/network_plots.py --help
```

---

### Q4: What if I have custom modifications to old scripts?

**A:** 

1. Check if the functionality is available via parameters in new scripts
2. If not, the old script is in `scripts/legacy/` - you can still run it
3. Consider submitting a feature request for the new unified script

---

### Q5: How do I migrate my automation scripts?

**A:** Use the [Command-Line Examples](#command-line-examples) section as a template. Key changes:

1. Update script paths
2. Add method/mode parameters
3. Update import statements

Example migration:

```bash
# OLD automation script
for method in FASK GIMME MVAR MVGC PCMCI RASL; do
    python ${method}_fig4.py --output results/${method}.png
done

# NEW automation script  
python benchmarks/benchmark_runner.py \
    --method all \
    --output-dir results/ \
    --format png
```

---

### Q6: Are there performance differences?

**A:** No significant differences. The new scripts use the same underlying algorithms. Some improvements:

- **Faster imports**: Reduced redundant imports
- **Better parallelization**: Unified scripts can batch operations
- **Less disk I/O**: Shared result caching

---

### Q7: What if I find a bug?

**A:** 

1. Check if the bug exists in both old and new scripts (may be pre-existing)
2. Report with both old script name and new script name + parameters
3. Old scripts in `legacy/` are no longer maintained

---

### Q8: How do I reproduce old results exactly?

**A:** Old scripts are preserved in `scripts/legacy/` with original code intact. To use:

```bash
# Run old script from legacy
python scripts/legacy/experimental/MVAR_fig4_variant3.py

# Or copy to current directory
cp scripts/legacy/experimental/MVAR_fig4_variant3.py ./
python MVAR_fig4_variant3.py
```

---

### Q9: Can I contribute new features to unified scripts?

**A:** Yes! Unified scripts are designed to be extensible:

```python
# Example: Adding a new benchmark method
# Edit: benchmarks/benchmark_runner.py

SUPPORTED_METHODS = ['FASK', 'GIMME', 'MVAR', 'MVGC', 'PCMCI', 'RASL', 'YOUR_METHOD']

def run_your_method(config):
    # Implement your method
    pass
```

Submit pull requests with:
- New method implementation
- Unit tests
- Updated documentation

---

### Q10: Where is the documentation now?

**A:** Consolidated in `docs/`:

| Old Files (14+) | New File |
|-----------------|----------|
| `HYPERPARAMETER_TUNING_SUMMARY.md`<br>`COMPLETE_SUMMARY.md`<br>`VAR_HYPERPARAMETER_TUNING_README.md` | `docs/experiments/hyperparameter_tuning.md` |
| `QUICK_START.txt`<br>`FILE_STRUCTURE.txt`<br>`GETTING_STARTED.md` | `docs/QUICKSTART.md` |
| `CLUSTER_QUICK_START.txt`<br>`SLURM_GUIDE.md` | `docs/experiments/cluster_guide.md` |

---

## Getting Help

### Resources

- **Quick Start**: `docs/QUICKSTART.md`
- **API Reference**: `docs/API.md`
- **Examples**: `docs/examples/`
- **This Migration Guide**: `MIGRATION.md`

### Support Channels

1. **Check documentation first**: `docs/` directory
2. **Script help**: Run with `--help` flag
3. **Issues**: GitHub Issues with `migration` tag
4. **Questions**: GitHub Discussions

---

## Migration Checklist

Use this checklist to update your workflows:

- [ ] Read this migration guide completely
- [ ] Identify which old scripts you use (check your automation/cron jobs)
- [ ] Map old scripts to new scripts using Quick Reference table
- [ ] Test new scripts with `--help` to understand parameters
- [ ] Update your automation scripts/pipelines
- [ ] Update import statements in custom Python code
- [ ] Update documentation/README in your projects
- [ ] Test that results are equivalent (use `legacy/` for comparison)
- [ ] Update bookmarks/aliases in your shell
- [ ] Remove dependencies on deprecated wrapper scripts

---

## Timeline

| Date | Event |
|------|-------|
| **Dec 2025** | v2.0.0 released with refactoring |
| **Dec 2025 - Mar 2026** | Deprecation period (wrappers available) |
| **Mar 2026** | v2.1.0 removes deprecated wrappers |
| **Jun 2026** | `legacy/` scripts may be removed |

---

## Summary

### Key Takeaways

✅ **118+ scripts → 33 core modules**  
✅ **Logical organization** by function (analysis/benchmarks/experiments/visualization)  
✅ **Unified interfaces** with parameters instead of script duplication  
✅ **Backward compatibility** via temporary wrapper scripts  
✅ **Comprehensive documentation** in `docs/`

### Next Steps

1. **Read** `docs/QUICKSTART.md` for basic usage
2. **Explore** new script structure in `scripts/` folders
3. **Test** your workflows with new scripts
4. **Report** any issues or missing functionality

---

**Version**: 2.0.0  
**Last Updated**: December 2, 2025  
**Maintainer**: Gunfolds Development Team


