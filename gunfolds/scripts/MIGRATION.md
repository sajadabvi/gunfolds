# Migration Guide

This guide helps you transition from old scripts to the new unified codebase structure.

## Overview of Changes

The gunfolds codebase has been reorganized to improve maintainability, reduce redundancy, and provide a more professional structure. **~85 script files have been consolidated into ~10 unified modules.**

## Quick Reference

| Old Script | New Script | Notes |
|------------|------------|-------|
| `plot_fmri_enhanced.py` | `visualization/network_plots.py --source fmri` | Unified plotting |
| `plot_gcm_enhanced.py` | `visualization/network_plots.py --source gcm` | Unified plotting |
| `plot_manuscript_compact.py` | `visualization/network_plots.py --compact` | Unified plotting |
| `FASK_fig4.py` | `benchmarks/benchmark_runner.py --method FASK` | Unified benchmarking |
| `GIMME_fig4.py` | `benchmarks/benchmark_runner.py --method GIMME` | Unified benchmarking |
| `MVAR_fig4.py` | `benchmarks/benchmark_runner.py --method MVAR` | Unified benchmarking |
| `MVGC_fig4.py` | `benchmarks/benchmark_runner.py --method MVGC` | Unified benchmarking |
| `PCMCI_fig4.py` | `benchmarks/benchmark_runner.py --method PCMCI` | Unified benchmarking |
| `RASL_fig4.py` | `benchmarks/benchmark_runner.py --method RASL` | Unified benchmarking |
| `*_time_undersampling_data.py` | `benchmarks/time_undersampling.py` | Unified with --method flag |
| `VAR_analyze_delta.py` | `experiments/var_analyzer.py --analysis-type delta` | Unified analysis |
| `VAR_analyze_hyperparameters.py` | `experiments/var_analyzer.py --analysis-type hyperparameters` | Unified analysis |
| `VAR_analyze_orientation_only.py` | `experiments/var_analyzer.py --analysis-type orientation` | Unified analysis |

## Detailed Migration Instructions

### 1. Enhanced Plotting Scripts

**Old Way:**
```bash
python plot_gcm_enhanced.py -t 11272025173313
python plot_fmri_enhanced.py -t 11262025164900 --groups combined 0 1
python plot_manuscript_compact.py  # hardcoded timestamps
```

**New Way:**
```bash
python visualization/network_plots.py --source gcm --timestamp 11272025173313
python visualization/network_plots.py --source fmri --timestamp 11262025164900 --groups combined 0 1
python visualization/network_plots.py --source gcm --timestamp 11272025173313 --compact
```

**Key Changes:**
- Single unified script with `--source` parameter
- Consistent command-line interface
- All functionality preserved
- Added `--spring-layout` option for GCM

### 2. Benchmark (Fig 4) Scripts

**Old Way:**
```bash
python MVGC_fig4.py  # Runs MVGC, creates plots
python GIMME_fig4.py  # Runs GIMME, creates plots
# ... 6 separate scripts
```

**New Way:**
```bash
python benchmarks/benchmark_runner.py --method MVGC --networks 1 2 3 4 5 6
python benchmarks/benchmark_runner.py --method GIMME --networks 1 2 3 4 5 6 --baseline
python benchmarks/benchmark_runner.py --method MVAR --plot-only results/MVAR_benchmark.zkl
```

**Key Changes:**
- Single script handles all methods
- `--method` parameter selects which method to run
- `--baseline` adds comparison with reported results
- `--plot-only` for visualization without recomputing
- All 6 methods use identical interface

**Migration Steps:**
1. Replace method name in PreFix variable with `--method` argument
2. Results save to `results/[METHOD]_benchmark_TIMESTAMP.zkl`
3. Plots automatically generated

### 3. Time Undersampling Scripts

**Old Way:**
```bash
python MVGC_time_undersampling_data.py
python slurm_GIMME_time_undersampling_data.py  # Separate SLURM version
# ... 7 separate scripts
```

**New Way:**
```bash
# Local execution
python benchmarks/time_undersampling.py --method MVGC --networks 1 2 3

# SLURM execution
python benchmarks/time_undersampling.py --method GIMME --slurm --job-id 1
```

**Key Changes:**
- Single script for all methods
- `--slurm` flag enables cluster mode (no separate script needed)
- `--apply-rasl` option for post-processing
- Unified output format

### 4. VAR Analysis Scripts

**Old Way:**
```bash
python VAR_analyze_delta.py -f delta_results.csv
python VAR_analyze_hyperparameters.py -f hp_results.csv -o output/
python VAR_analyze_orientation_only.py -f orientation_results.csv
```

**New Way:**
```bash
python experiments/var_analyzer.py --analysis-type delta -f delta_results.csv
python experiments/var_analyzer.py --analysis-type hyperparameters -f hp_results.csv -o output/
python experiments/var_analyzer.py --analysis-type orientation -f orientation_results.csv
```

**Key Changes:**
- Single script with `--analysis-type` parameter
- All three analysis modes available
- Consistent output format
- Shared plotting utilities

### 5. Linear Statistics Scripts

**Old Way:**
```bash
python lineal_stat_scan.py -n 6 -d 0.25
python lineal_stat_scan2.py -n 6 -d 0.15  # Different defaults
python linear_stat_continious_weights.py -n 6
# ... 9 variants
```

**New Way:**
```bash
python simulation/linear_stat_scanner.py -n 6 -d 0.25 --scan-type basic
python simulation/linear_stat_scanner.py -n 6 -d 0.15 --scan-type config_test
python simulation/linear_stat_scanner.py -n 6 --continuous-weights
```

**Key Changes:**
- Single script with `--scan-type` parameter
- `--continuous-weights` flag for weight variants
- Unified parameter handling
- All variants accessible

### 6. Weighted Solver Scripts

**Old Way:**
```bash
python weighted.py -n 5 -u 3 -d 1.5
python weighted_optN.py -n 5 -u 3 -d 1.5  # Adds GT_density
python weighted_then_drasl.py -n 5 -u 3 -d 1.5  # Two-stage
```

**New Way:**
```bash
python simulation/weighted_solver.py -n 5 -u 3 -d 1.5
python simulation/weighted_solver.py -n 5 -u 3 -d 1.5 --use-gt-density
python simulation/weighted_solver.py -n 5 -u 3 -d 1.5 --two-stage
```

**Key Changes:**
- Single script with feature flags
- `--use-gt-density` for optN variant
- `--two-stage` for DRASL post-processing

## Breaking Changes

### File Locations

Old scripts in `gunfolds/scripts/` have moved to subdirectories:
- `analysis/` - Result analysis
- `benchmarks/` - Method comparisons  
- `experiments/` - VAR experiments
- `visualization/` - Plotting
- `simulation/` - Simulations
- `real_data/` - Real datasets
- `legacy/` - Old scripts (archived)

**Update imports:**
```python
# Old
from gunfolds.scripts import my_functions as mf

# New
from gunfolds.scripts.utils import common_functions as cf
```

### Command-Line Arguments

Many scripts now use consistent argument names:

| Old | New | Reason |
|-----|-----|--------|
| `-t` | `--timestamp` | More descriptive |
| Various | `--method` | Unified parameter |
| N/A | `--source` | New for unified plotting |
| N/A | `--analysis-type` | New for unified analysis |

### Output Formats

- All results save to `results/` by default (configurable with `--output-dir`)
- Individual job outputs save to `results/individual_jobs/`
- Plots save alongside data files
- Consistent `.zkl` format for results

## Import Changes

### Common Functions

**Old:**
```python
from gunfolds.scripts import my_functions as mf

mf.precision_recall(est, gt)
mf.find_two_cycles(graph)
mf.calculate_f1_score(p, r)
```

**New:**
```python
from gunfolds.scripts.utils import common_functions as cf

cf.precision_recall(est, gt)
cf.find_two_cycles(graph)
cf.calculate_f1_score(p, r)
```

### Network Plots

**Old:**
```python
# Functionality embedded in individual scripts
```

**New:**
```python
from gunfolds.scripts.visualization import network_plots as np

np.plot_network_circular(edge_rate, threshold=0.5, output_path='out.png')
np.plot_heatmap_enhanced(edge_rate, output_path='heatmap.png')
```

## Workflow Updates

### Before: Multiple Similar Scripts

```bash
# Had to maintain 6 nearly-identical scripts
vim MVGC_fig4.py
vim MVAR_fig4.py
vim GIMME_fig4.py
# ... fix bug 6 times
```

### After: Single Parameterized Script

```bash
# Fix once, works for all methods
vim benchmarks/benchmark_runner.py
# Run for all methods
for method in MVGC MVAR GIMME FASK PCMCI RASL; do
    python benchmarks/benchmark_runner.py --method $method
done
```

## Backward Compatibility

### Transition Period

Old scripts are in `legacy/` folder with deprecation warnings:
- Still functional for reference
- Will be removed in future release
- **Migrate as soon as possible**

### Using Legacy Scripts

```bash
# Still works, but shows warning
python legacy/plot_gcm_enhanced.py -t 11272025173313

# Output:
# WARNING: This script is deprecated. Use:
#   python visualization/network_plots.py --source gcm --timestamp 11272025173313
```

## Documentation Updates

### Old Documentation

Multiple overlapping files in `scripts/`:
- `HYPERPARAMETER_TUNING_SUMMARY.md`
- `COMPLETE_SUMMARY.md`
- `QUICK_START.txt`
- `CLUSTER_QUICK_START.txt`
- ... 10+ more files

### New Documentation

Consolidated into 3 main files in `docs/`:
- `docs/QUICKSTART.md` - Getting started guide
- `docs/experiments/hyperparameter_tuning.md` - Detailed tuning guide
- `docs/experiments/cluster_guide.md` - Cluster execution guide

## Troubleshooting

### Issue: Can't find old script

**Solution:** Check `legacy/README.md` for mapping to new script

### Issue: Import errors after migration

**Solution:** Update imports from `my_functions` to `common_functions`

### Issue: Different output format

**Solution:** New scripts save to `results/` by default. Use `--output-dir` to customize.

### Issue: Missing features from old script

**Solution:** Check new script's `--help`. Most features preserved with flags.

## Getting Help

1. **Check unified script help:**
   ```bash
   python benchmarks/benchmark_runner.py --help
   ```

2. **Review examples in docs:**
   - `docs/QUICKSTART.md`
   - `docs/experiments/`

3. **Look at legacy script:**
   - Still in `legacy/` for reference
   - Compare old vs new usage

4. **File an issue:**
   - If functionality is missing
   - If migration is unclear

## Checklist for Migration

- [ ] Update any hardcoded script paths in workflows
- [ ] Replace script calls with new unified versions
- [ ] Update imports from `my_functions` to `common_functions`
- [ ] Test new scripts produce equivalent results
- [ ] Update documentation/README files
- [ ] Remove dependencies on legacy scripts
- [ ] Update SLURM submission scripts
- [ ] Verify output paths still work with downstream tools

## Timeline

- **Current:** Both old and new scripts available
- **After 1 month:** Legacy scripts show deprecation warnings
- **After 3 months:** Legacy scripts may be removed
- **Goal:** All users migrated to new structure

## Benefits of Migration

✅ **Fewer files** - 118 scripts → ~10 unified modules
✅ **Easier maintenance** - Fix bugs once, not 6 times  
✅ **Consistent interface** - Same arguments across methods
✅ **Better documentation** - Consolidated guides
✅ **Professional structure** - Clear organization

## Questions?

See:
- `docs/QUICKSTART.md` for usage examples
- `legacy/README.md` for old → new mapping
- `docs/experiments/` for detailed guides

