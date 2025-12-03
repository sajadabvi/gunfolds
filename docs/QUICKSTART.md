# Gunfolds Quick Start Guide

## Overview

Gunfolds is a Python package for causal discovery in undersampled time series data using the FCI* algorithm and related methods.

## Installation

```bash
git clone https://github.com/[your-repo]/gunfolds.git
cd gunfolds
pip install -e .
```

## Basic Usage

### 1. Running Benchmarks

Compare different causal discovery methods on simulated networks:

```bash
# Run MVGC benchmark on networks 1-3
python gunfolds/scripts/benchmarks/benchmark_runner.py --method MVGC --networks 1 2 3

# Run with baseline comparison
python gunfolds/scripts/benchmarks/benchmark_runner.py --method GIMME --networks 1 2 3 4 5 6 --baseline

# Only generate plots from saved results
python gunfolds/scripts/benchmarks/benchmark_runner.py --method MVAR --plot-only results/MVAR_benchmark.zkl
```

**Available methods:** MVGC, MVAR, FASK, GIMME, PCMCI, RASL

### 2. Network Visualization

Create publication-quality network plots:

```bash
# Visualize GCM results
python gunfolds/scripts/visualization/network_plots.py --source gcm --timestamp 11272025173313

# Visualize fMRI/RASL results
python gunfolds/scripts/visualization/network_plots.py --source fmri --timestamp 11262025164900 --groups combined 0 1

# Create compact manuscript figures
python gunfolds/scripts/visualization/network_plots.py --source gcm --timestamp 11272025173313 --compact
```

### 3. Hyperparameter Tuning

Optimize RASL priority parameters:

```bash
# Run hyperparameter tuning (subset mode, ~20 configs)
cd gunfolds/scripts
./run_hyperparameter_tuning.sh --subset

# Analyze results
python experiments/var_analyzer.py --analysis-type hyperparameters -f results/priority_tuning_*.csv
```

### 4. Time Undersampling Experiments

Test methods under different undersampling rates:

```bash
# Local execution
python gunfolds/scripts/benchmarks/time_undersampling.py --method MVGC --networks 1 2 3

# SLURM cluster execution (single job)
python gunfolds/scripts/benchmarks/time_undersampling.py --method GIMME --slurm --job-id 1
```

## Common Workflows

### Workflow 1: Method Comparison

```bash
# 1. Run benchmarks for multiple methods
for method in MVGC MVAR GIMME; do
    python gunfolds/scripts/benchmarks/benchmark_runner.py --method $method
done

# 2. Compare results
# Results saved to: results/[METHOD]_benchmark_*.zkl
```

### Workflow 2: Real Data Analysis

```bash
# 1. Run GCM on fMRI data
python gunfolds/scripts/real_data/roebroeck_gcm.py

# 2. Visualize results
python gunfolds/scripts/visualization/network_plots.py --source gcm --timestamp [TIMESTAMP]

# 3. Run RASL on fMRI data
python gunfolds/scripts/real_data/fmri_experiment.py

# 4. Visualize RASL results
python gunfolds/scripts/visualization/network_plots.py --source fmri --timestamp [TIMESTAMP]
```

### Workflow 3: Parameter Optimization

```bash
# 1. Test hyperparameter configurations
cd gunfolds/scripts
./run_hyperparameter_tuning.sh --subset

# 2. Analyze which parameters work best
python experiments/var_analyzer.py --analysis-type hyperparameters -f results/priority_tuning_*.csv

# 3. Test delta tuning (solution selection)
python experiments/var_delta_tuning.py -n 3 -u 3

# 4. Analyze delta results
python experiments/var_analyzer.py --analysis-type delta -f results/delta_tuning_*.csv
```

## Directory Structure

```
gunfolds/
├── scripts/
│   ├── analysis/              # Result analysis tools
│   ├── benchmarks/            # Method comparison experiments
│   ├── experiments/           # VAR experiments & tuning
│   ├── visualization/         # Plotting functions
│   ├── simulation/            # Simulation & data generation
│   ├── real_data/             # Real dataset experiments
│   ├── cluster/               # SLURM/cluster scripts
│   ├── legacy/                # Archived/deprecated scripts
│   └── utils/                 # Shared utilities
├── gunfolds/                  # Core package
│   ├── estimation/            # Causal discovery algorithms
│   ├── solvers/               # RASL solvers
│   ├── utils/                 # Utility functions
│   └── viz/                   # Visualization tools
└── docs/                      # Documentation
    ├── QUICKSTART.md          # This file
    └── experiments/           # Experiment-specific guides
```

## Getting Help

### Command-Line Help

All scripts support `--help`:

```bash
python gunfolds/scripts/benchmarks/benchmark_runner.py --help
python gunfolds/scripts/visualization/network_plots.py --help
python gunfolds/scripts/experiments/var_analyzer.py --help
```

### Documentation

- **Hyperparameter Tuning:** See `docs/experiments/hyperparameter_tuning.md`
- **Cluster Execution:** See `docs/experiments/cluster_guide.md`
- **Migration Guide:** See `gunfolds/scripts/MIGRATION.md`

### Common Issues

**Issue:** ImportError for py_tetrad or tigramite
- **Solution:** Install optional dependencies: `pip install py-tetrad tigramite`

**Issue:** No module named 'gunfolds'
- **Solution:** Install package: `pip install -e .` from repository root

**Issue:** SLURM jobs fail
- **Solution:** Check paths in cluster scripts and ensure data files exist

## Next Steps

1. **Run a test benchmark:** Try `benchmark_runner.py` with a small network
2. **Visualize results:** Use `network_plots.py` on your benchmark output
3. **Optimize parameters:** Run hyperparameter tuning if needed
4. **Process real data:** Use scripts in `real_data/` for your datasets

For detailed experiment guides, see `docs/experiments/`.
