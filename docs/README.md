# Gunfolds Documentation

**Version 2.0.0** - Complete Refactoring & Reorganization

---

## 📚 Documentation Overview

This directory contains comprehensive documentation for the refactored gunfolds codebase.

### Quick Links

| Document | Purpose | Audience |
|----------|---------|----------|
| **[QUICKSTART.md](QUICKSTART.md)** | Get started quickly with common tasks | All users |
| **[MIGRATION.md](../MIGRATION.md)** | Migrate from old codebase to new structure | Existing users |
| **[OLD_TO_NEW_SCRIPT_REFERENCE.md](../OLD_TO_NEW_SCRIPT_REFERENCE.md)** | Quick lookup table for old scripts | Existing users |
| **[experiments/hyperparameter_tuning.md](experiments/hyperparameter_tuning.md)** | Comprehensive tuning guide | Advanced users |
| **[experiments/cluster_guide.md](experiments/cluster_guide.md)** | SLURM/cluster computing | Cluster users |

---

## 🎯 New User? Start Here

### 1. Install Gunfolds

```bash
pip install gunfolds
```

Or from source:

```bash
git clone https://github.com/your-org/gunfolds.git
cd gunfolds
pip install -e .
```

### 2. Read the Quick Start

Start with **[QUICKSTART.md](QUICKSTART.md)** to learn:
- Basic concepts
- Common tasks
- Script organization
- Your first complete workflow

### 3. Explore Examples

Try the example workflows in the Quick Start guide, then check `examples/` for more advanced use cases.

---

## 🔄 Migrating from Old Codebase?

### Step-by-Step Migration

1. **Read [MIGRATION.md](../MIGRATION.md)** - Comprehensive migration guide with:
   - Overview of changes
   - Detailed category-by-category migration
   - Command-line examples
   - Breaking changes
   - FAQ

2. **Use [OLD_TO_NEW_SCRIPT_REFERENCE.md](../OLD_TO_NEW_SCRIPT_REFERENCE.md)** - Quick lookup:
   - Complete A-Z listing of old scripts
   - Direct mapping to new scripts
   - Migration commands
   - Detailed examples

3. **Test Your Workflows** - Verify that results match:
   ```bash
   # Run old script (from legacy/)
   python scripts/legacy/experimental/OLD_SCRIPT.py
   
   # Run new equivalent
   python scripts/NEW_CATEGORY/new_script.py --method METHOD
   
   # Compare results
   diff -r results_old/ results_new/
   ```

### Migration Quick Reference

```bash
# Benchmarks (fig4 scripts)
python [METHOD]_fig4.py  →  python benchmarks/benchmark_runner.py --method [METHOD]

# Enhanced plotting
python plot_[TYPE]_enhanced.py  →  python visualization/network_plots.py --data-source [TYPE]

# Time undersampling
python [METHOD]_time_undersampling_data.py  →  python benchmarks/time_undersampling.py --method [METHOD]

# SLURM jobs
python slurm_*.py  →  Add --slurm flag to corresponding script

# VAR analysis
python VAR_analyze_[TYPE].py  →  python experiments/var_analyzer.py --analysis-type [TYPE]
```

---

## 📖 Documentation by Topic

### Getting Started

- **[QUICKSTART.md](QUICKSTART.md)** - Installation, basic usage, first analysis
- **README.md** (this file) - Documentation overview

### Migration & Reference

- **[MIGRATION.md](../MIGRATION.md)** - Complete migration guide
- **[OLD_TO_NEW_SCRIPT_REFERENCE.md](../OLD_TO_NEW_SCRIPT_REFERENCE.md)** - Script lookup table

### Advanced Topics

- **[experiments/hyperparameter_tuning.md](experiments/hyperparameter_tuning.md)**
  - VAR model parameters
  - RASL solver parameters
  - Systematic tuning workflows
  - Example tuning studies
  
- **[experiments/cluster_guide.md](experiments/cluster_guide.md)**
  - SLURM basics
  - Submitting jobs
  - Monitoring and managing jobs
  - Best practices
  - Common use cases

---

## 🗂️ New Directory Structure

The refactored codebase is organized by function:

```
gunfolds/
├── scripts/
│   ├── analysis/              # Result analysis tools
│   │   └── result_analyzer.py
│   ├── benchmarks/            # Method comparison experiments
│   │   ├── benchmark_runner.py
│   │   ├── time_undersampling.py
│   │   └── multi_rasl_runner.py
│   ├── experiments/           # VAR experiments & tuning
│   │   ├── var_analyzer.py
│   │   └── var_hyperparameter_tuning.py
│   ├── visualization/         # Plotting and figures
│   │   ├── network_plots.py
│   │   └── manuscript_figures.py
│   ├── simulation/            # Data generation
│   │   ├── linear_stat_scanner.py
│   │   └── weighted_solver.py
│   ├── real_data/             # Real dataset experiments
│   │   ├── fmri_experiment.py
│   │   └── roebroeck_gcm.py
│   ├── cluster/               # SLURM/cluster scripts
│   │   └── job_submission.sh
│   └── legacy/                # Archived old code
│       ├── archive_old/
│       ├── experimental/
│       ├── testing/
│       └── variants/
└── docs/                      # Documentation (this directory)
    ├── README.md              # This file
    ├── QUICKSTART.md
    └── experiments/
        ├── hyperparameter_tuning.md
        └── cluster_guide.md
```

### What Goes Where?

| I want to... | Use this directory |
|--------------|-------------------|
| Analyze results | `scripts/analysis/` |
| Compare methods | `scripts/benchmarks/` |
| Tune hyperparameters | `scripts/experiments/` |
| Create plots | `scripts/visualization/` |
| Generate simulation data | `scripts/simulation/` |
| Analyze real data | `scripts/real_data/` |
| Submit cluster jobs | `scripts/cluster/` |
| Find old scripts | `scripts/legacy/` |

---

## 🚀 Common Workflows

### Workflow 1: Complete Analysis Pipeline

```bash
# 1. Generate simulation data
python scripts/simulation/linear_stat_scanner.py \
    --nodes 5,7,10 \
    --samples 100 \
    --output-dir results/simulations/

# 2. Run benchmarks
python scripts/benchmarks/benchmark_runner.py \
    --method all \
    --input-dir results/simulations/ \
    --output-dir results/benchmarks/

# 3. Analyze results
python scripts/analysis/result_analyzer.py \
    --data-type simulation \
    --results-dir results/benchmarks/ \
    --output results/summary.csv

# 4. Create visualizations
python scripts/visualization/network_plots.py \
    --data-source rasl \
    --input results/benchmarks/RASL_results.pkl \
    --output figures/network.pdf
```

### Workflow 2: Hyperparameter Tuning

```bash
# 1. Define parameter grid (create config file)
cat > tuning_config.yaml <<EOF
var_parameters:
  orders: [1, 2, 3, 4]
  lambda: [0.01, 0.1, 1.0]
  thresholds: [0.05, 0.1, 0.15]
EOF

# 2. Run grid search
python scripts/experiments/var_hyperparameter_tuning.py \
    --config tuning_config.yaml \
    --cv-folds 5 \
    --output-dir results/tuning/

# 3. Analyze results
python scripts/experiments/var_analyzer.py \
    --analysis-type hyperparameters \
    --input results/tuning/results.csv \
    --output-dir results/analysis/

# 4. Use best config
python scripts/benchmarks/benchmark_runner.py \
    --method RASL \
    --config results/analysis/best_config.yaml
```

### Workflow 3: Large-Scale Cluster Study

```bash
# 1. Submit batch job to cluster
python scripts/benchmarks/time_undersampling.py \
    --method all \
    --nodes 5,10,15,20 \
    --samples 1000 \
    --undersampling 2,4,8 \
    --slurm \
    --submit \
    --partition normal \
    --time 02:00:00 \
    --mem 16G

# 2. Monitor jobs
squeue -u $USER

# 3. Collect results when complete
python scripts/analysis/result_analyzer.py \
    --data-type simulation \
    --results-dir $SCRATCH/gunfolds_results/ \
    --output cluster_summary.csv
```

---

## 💡 Key Improvements in v2.0

### Before (v1.x)

❌ **118+ scripts** with massive duplication  
❌ **6 identical fig4 scripts** (350 lines each)  
❌ **14+ overlapping documentation files**  
❌ **No clear organization** (all scripts in one folder)  
❌ **Hardcoded parameters** (need new script for each variant)

### After (v2.0)

✅ **~33 core scripts** (85 fewer files!)  
✅ **Unified benchmark runner** (1 script replaces 6)  
✅ **Consolidated documentation** (4 main guides)  
✅ **Logical folder structure** (organized by function)  
✅ **Parameterized scripts** (command-line flags for variants)

### Benefits

- **Easier maintenance**: Fix bugs once, not 6 times
- **Better discoverability**: Logical folder structure
- **Faster development**: Shared utilities, less duplication
- **Professional appearance**: Organized, well-documented
- **Backward compatibility**: Old scripts preserved in `legacy/`

---

## 📊 Script Categories Explained

### Analysis Scripts (`scripts/analysis/`)

**Purpose**: Analyze results from experiments

**Main script**: `result_analyzer.py`

```bash
# Analyze GCM results
python scripts/analysis/result_analyzer.py --data-type gcm --results-dir DIR

# Analyze simulation results
python scripts/analysis/result_analyzer.py --data-type simulation --results-dir DIR

# Quick analysis mode
python scripts/analysis/result_analyzer.py --quick --results-dir DIR
```

---

### Benchmark Scripts (`scripts/benchmarks/`)

**Purpose**: Compare different causal discovery methods

**Main scripts**:
- `benchmark_runner.py` - Method comparison (replaces 6 fig4 scripts)
- `time_undersampling.py` - Undersampling rate experiments (replaces 8 scripts)
- `multi_rasl_runner.py` - Multiple RASL runs (replaces 6 scripts)

```bash
# Compare all methods
python scripts/benchmarks/benchmark_runner.py --method all

# Test undersampling rates
python scripts/benchmarks/time_undersampling.py --method RASL --undersampling 2,4,8

# Multiple RASL experiments
python scripts/benchmarks/multi_rasl_runner.py --method MVAR --mode individual
```

---

### Experiment Scripts (`scripts/experiments/`)

**Purpose**: VAR hyperparameter tuning and experiments

**Main scripts**:
- `var_analyzer.py` - Analyze VAR results (replaces 3 scripts)
- `var_hyperparameter_tuning.py` - Systematic parameter tuning

```bash
# Analyze VAR hyperparameters
python scripts/experiments/var_analyzer.py --analysis-type hyperparameters

# Run hyperparameter grid search
python scripts/experiments/var_hyperparameter_tuning.py --config config.yaml
```

---

### Visualization Scripts (`scripts/visualization/`)

**Purpose**: Create plots and figures for manuscripts

**Main scripts**:
- `network_plots.py` - Network visualizations (replaces 3 enhanced plotting scripts)
- `manuscript_figures.py` - Publication-quality figures

```bash
# Create network plot
python scripts/visualization/network_plots.py --data-source fmri --output fig.pdf

# Generate all manuscript figures
python scripts/visualization/manuscript_figures.py --figure all
```

---

### Simulation Scripts (`scripts/simulation/`)

**Purpose**: Generate synthetic data for experiments

**Main scripts**:
- `linear_stat_scanner.py` - Scan linear statistics (replaces 9 scripts)
- `weighted_solver.py` - Weighted solver variants (replaces 3 scripts)

```bash
# Linear statistics scan
python scripts/simulation/linear_stat_scanner.py --nodes 5,10,15

# Weighted solver
python scripts/simulation/weighted_solver.py --nodes 10 --optimize-density
```

---

### Real Data Scripts (`scripts/real_data/`)

**Purpose**: Analyze real datasets (fMRI, GCM, macaque)

**Scripts** (moved from root, not merged):
- `fmri_experiment.py`
- `roebroeck_gcm.py`
- `macaque_analysis.py`

```bash
python scripts/real_data/fmri_experiment.py --input data/fmri.csv
python scripts/real_data/roebroeck_gcm.py --input data/gcm.mat
```

---

## 🛠️ Getting Help

### Script-Specific Help

All scripts have comprehensive `--help`:

```bash
python scripts/benchmarks/benchmark_runner.py --help
python scripts/visualization/network_plots.py --help
python scripts/experiments/var_analyzer.py --help
```

### Documentation

- **Getting Started**: [QUICKSTART.md](QUICKSTART.md)
- **Migration**: [MIGRATION.md](../MIGRATION.md)
- **Script Lookup**: [OLD_TO_NEW_SCRIPT_REFERENCE.md](../OLD_TO_NEW_SCRIPT_REFERENCE.md)
- **Hyperparameter Tuning**: [experiments/hyperparameter_tuning.md](experiments/hyperparameter_tuning.md)
- **Cluster Computing**: [experiments/cluster_guide.md](experiments/cluster_guide.md)

### Community Support

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, workflow sharing
- **Email**: gunfolds-dev@example.com (if applicable)

---

## 📝 Contributing

### Reporting Issues

Found a bug or missing functionality?

1. Check if it's a known issue in [GitHub Issues](https://github.com/your-org/gunfolds/issues)
2. For migration issues, include:
   - Old script name
   - New script command you tried
   - Expected vs. actual behavior
3. For bugs in new scripts, include:
   - Script name and command
   - Error message
   - Minimal reproducible example

### Suggesting Improvements

Want to suggest a feature or improvement?

1. Open a GitHub Discussion
2. Describe your use case
3. Propose a solution (if you have one)

### Contributing Code

We welcome contributions! To add features:

1. Fork the repository
2. Create a feature branch
3. Add your feature with tests and documentation
4. Submit a pull request

See `CONTRIBUTING.md` (if available) for detailed guidelines.

---

## 📚 Additional Resources

### Research Papers

Gunfolds implements methods from these papers:

- **RASL**: Hyttinen et al. (2013), "Learning Linear Cyclic Causal Models with Latent Variables"
- **dRASL**: Malinsky & Danks (2018), "Causal Discovery Algorithms: A Practical Guide"
- **Undersampling**: Danks & Patel (2018), "Undersampling in Causal Discovery"

### Related Tools

- **CausalNex**: Bayesian networks in Python
- **Tigramite**: Time series causal discovery
- **pcalg**: R package for PC algorithm

### Tutorials

- Basic causal discovery: [link]
- Advanced RASL usage: [link]
- Real data analysis: [link]

---

## 🔄 Version History

### v2.0.0 (December 2025)

**Major refactoring**:
- Consolidated 118+ scripts → 33 core modules
- New logical folder structure
- Unified documentation
- Backward compatibility via `legacy/` folder

### v1.x (Previous versions)

- Original script organization
- Scripts now in `scripts/legacy/`

---

## ❓ FAQ

### Q: Where did my script go?

**A**: Check [OLD_TO_NEW_SCRIPT_REFERENCE.md](../OLD_TO_NEW_SCRIPT_REFERENCE.md) for a complete mapping.

### Q: Can I still use old scripts?

**A**: Yes, they're preserved in `scripts/legacy/` and will work as before.

### Q: Are results identical between old and new scripts?

**A**: Yes, the underlying algorithms are unchanged. New scripts use the same code with better organization.

### Q: How do I report missing functionality?

**A**: Open a GitHub Issue with the tag `migration` and describe what's missing.

### Q: When will legacy scripts be removed?

**A**: Not planned for removal before June 2026. See [MIGRATION.md](../MIGRATION.md) for timeline.

---

## 📬 Contact

- **GitHub**: https://github.com/your-org/gunfolds
- **Issues**: https://github.com/your-org/gunfolds/issues
- **Discussions**: https://github.com/your-org/gunfolds/discussions
- **Email**: gunfolds-dev@example.com (if applicable)

---

**Happy analyzing!**

*Documentation last updated: December 2, 2025*  
*Gunfolds version: 2.0.0*  
*Maintained by: Gunfolds Development Team*

