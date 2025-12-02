# Gunfolds Scripts Directory

This directory contains organized, unified scripts for running gunfolds experiments and analyses.

## 🎯 Quick Start

```bash
# Run a benchmark
python benchmarks/benchmark_runner.py --method MVGC --networks 1 2 3

# Create visualizations
python visualization/network_plots.py --source gcm --timestamp TIMESTAMP

# Analyze hyperparameters
python experiments/var_analyzer.py --analysis-type hyperparameters -f results.csv
```

## 📁 Directory Structure

```
scripts/
├── analysis/              # Result analysis tools
│   └── result_analyzer.py         # Unified result analysis (GCM, RASL, simulations)
│
├── benchmarks/            # Method comparison experiments  
│   ├── benchmark_runner.py        # Compare methods (MVGC, MVAR, FASK, GIMME, PCMCI, RASL)
│   ├── time_undersampling.py      # Test undersampling rates
│   └── multi_rasl_runner.py       # Multi-individual RASL experiments
│
├── experiments/           # VAR experiments & hyperparameter tuning
│   ├── var_analyzer.py            # Analyze delta/hyperparameter/orientation results
│   ├── var_hyperparameter_tuning.py
│   ├── var_delta_tuning.py
│   ├── var_collector.py           # Collect parallel results
│   └── var_ruben_experiments.py
│
├── visualization/         # Publication-quality plotting
│   ├── network_plots.py           # Unified network visualization (GCM, fMRI, RASL)
│   ├── manuscript_figures.py
│   └── plotting_utils.py          # Shared plotting functions
│
├── simulation/            # Simulation & data generation
│   ├── linear_stat_scanner.py     # Unified linear statistics scanning
│   ├── weighted_solver.py         # Unified weighted RASL solver
│   └── data_generator.py
│
├── real_data/             # Real dataset experiments
│   ├── fmri_experiment.py         # fMRI RASL experiments
│   ├── roebroeck_gcm.py           # GCM on Roebroeck data
│   └── macaque_analysis.py        # Macaque data analysis
│
├── cluster/               # SLURM/cluster execution scripts
│   ├── slurm_templates.py
│   └── job_submission.sh
│
├── utils/                 # Shared utility functions
│   └── common_functions.py        # Extracted from my_functions.py
│
└── legacy/                # Archived/deprecated scripts
    └── README.md                   # Mapping old → new scripts
```

## 🔥 Common Tasks

### Run Benchmarks

```bash
# Single method
python benchmarks/benchmark_runner.py --method MVGC --networks 1 2 3 4 5 6

# With baseline comparison
python benchmarks/benchmark_runner.py --method GIMME --baseline

# Plot existing results
python benchmarks/benchmark_runner.py --method MVAR --plot-only results/MVAR_benchmark.zkl
```

### Visualize Results

```bash
# GCM results
python visualization/network_plots.py --source gcm --timestamp 11272025173313

# fMRI/RASL results (all groups)
python visualization/network_plots.py --source fmri --timestamp 11262025164900 --groups combined 0 1

# Manuscript figures (compact style)
python visualization/network_plots.py --source gcm --timestamp 11272025173313 --compact
```

### Hyperparameter Tuning

```bash
# Local: test subset (~20 configs, 1-2 hours)
cd /path/to/scripts
./run_hyperparameter_tuning.sh --subset

# Cluster: all 3125 configs (1-3 hours parallel)
sbatch --array=1-3125 slurm_hyperparameter_tuning.sh

# Analyze results
python experiments/var_analyzer.py --analysis-type hyperparameters -f results/priority_tuning_*.csv
```

### Time Undersampling Experiments

```bash
# Local execution
python benchmarks/time_undersampling.py --method MVGC --networks 1 2 3

# Cluster execution
python benchmarks/time_undersampling.py --method GIMME --slurm --job-id $SLURM_ARRAY_TASK_ID
```

## 📚 Documentation

- **Quick Start:** See [`../../docs/QUICKSTART.md`](../../docs/QUICKSTART.md)
- **Hyperparameter Tuning:** See [`../../docs/experiments/hyperparameter_tuning.md`](../../docs/experiments/hyperparameter_tuning.md)
- **Cluster Guide:** See [`../../docs/experiments/cluster_guide.md`](../../docs/experiments/cluster_guide.md)
- **Migration Guide:** See [`MIGRATION.md`](MIGRATION.md) for transitioning from old scripts

## 🔄 Migration from Old Scripts

If you're looking for an old script, check:

1. **Migration guide:** [`MIGRATION.md`](MIGRATION.md) has complete mapping
2. **Legacy folder:** [`legacy/README.md`](legacy/README.md) lists all archived scripts
3. **Quick reference:**
   - `plot_gcm_enhanced.py` → `visualization/network_plots.py --source gcm`
   - `*_fig4.py` → `benchmarks/benchmark_runner.py --method [METHOD]`
   - `VAR_analyze_*.py` → `experiments/var_analyzer.py --analysis-type [TYPE]`

## 💡 Key Improvements

### Before Reorganization
- 118 script files (many nearly identical)
- Redundant code across multiple files
- Inconsistent interfaces
- Scattered documentation

### After Reorganization
- ~10 unified modules
- Single source of truth for each function
- Consistent command-line interfaces
- Consolidated documentation

### Example: Benchmark Scripts

**Before:** 6 separate scripts (`MVGC_fig4.py`, `MVAR_fig4.py`, etc.)
```bash
python MVGC_fig4.py  # 350 lines
python MVAR_fig4.py  # 350 lines (95% identical)
# ... 4 more nearly-identical files
```

**After:** 1 unified script
```bash
python benchmarks/benchmark_runner.py --method MVGC
python benchmarks/benchmark_runner.py --method MVAR
# Same script, different parameter
```

## 🛠️ Development Guidelines

### Adding New Features

1. **Extend existing unified scripts** rather than creating new files
2. **Use command-line arguments** for variations
3. **Share common code** via `utils/common_functions.py`
4. **Document in appropriate guide** (`docs/QUICKSTART.md` or `docs/experiments/`)

### Code Style

- **Type hints** for function parameters
- **Docstrings** with Args/Returns sections
- **Argparse** for command-line interfaces
- **Consistent** naming and structure

### Testing

Before committing:
1. Test with `--help` to verify argument parsing
2. Run on small dataset to verify functionality
3. Check output format is consistent
4. Update relevant documentation

## 📞 Support

### Getting Help

1. **Command-line help:** All scripts support `--help`
   ```bash
   python benchmarks/benchmark_runner.py --help
   ```

2. **Documentation:** Check [`../../docs/`](../../docs/)

3. **Examples:** See [`../../docs/QUICKSTART.md`](../../docs/QUICKSTART.md)

4. **Migration:** See [`MIGRATION.md`](MIGRATION.md)

### Common Issues

**"Can't find old script"**
→ Check [`legacy/README.md`](legacy/README.md) for new location

**"Import errors"**
→ Update from `my_functions` to `utils/common_functions`

**"Different output format"**
→ New scripts save to `results/` by default

## 📊 Statistics

- **Scripts consolidated:** 85+ → 10 core modules
- **Code deduplication:** ~15,000 lines of redundant code removed
- **Documentation:** 14+ files → 3 comprehensive guides
- **Maintenance:** Fix bugs once instead of 6+ times

## 🎉 Benefits

✅ **Easier to use** - Consistent interfaces across all methods
✅ **Easier to maintain** - Single source for each feature
✅ **Better documented** - Comprehensive guides
✅ **More professional** - Clean, organized structure
✅ **Future-proof** - Easy to extend and enhance

---

**Version:** 2.0 (Post-Reorganization)
**Last Updated:** December 2024
**Maintainers:** See main repository

