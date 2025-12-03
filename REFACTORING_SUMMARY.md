# Gunfolds v2.0 Refactoring Summary

**Complete Documentation Package for Migration**

---

## 📦 Documentation Package Overview

This refactoring includes a complete documentation package to help you transition from the old codebase to the new organized structure.

### All Documentation Files

| File | Purpose | Who Should Read |
|------|---------|-----------------|
| **[MIGRATION.md](MIGRATION.md)** | Complete migration guide with detailed examples | All existing users |
| **[OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md)** | A-Z lookup table of old → new scripts | Quick reference for existing users |
| **[docs/README.md](docs/README.md)** | Documentation hub with links to all guides | All users |
| **[docs/QUICKSTART.md](docs/QUICKSTART.md)** | Getting started guide for new and existing users | All users |
| **[docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)** | Comprehensive hyperparameter tuning guide | Advanced users |
| **[docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)** | SLURM/cluster computing guide | Cluster users |
| **[code.plan.md](code.plan.md)** | Original refactoring plan (technical details) | Developers |
| **This file** | Summary and quick navigation | All users |

---

## 🚀 Quick Start Guide

### For New Users

```bash
# 1. Install
pip install gunfolds

# 2. Read the Quick Start
Start → docs/QUICKSTART.md

# 3. Try an example workflow
python scripts/benchmarks/benchmark_runner.py --method RASL --help
```

### For Existing Users Migrating

```bash
# 1. Read migration overview
Start → MIGRATION.md

# 2. Find your script
Look up → OLD_TO_NEW_SCRIPT_REFERENCE.md

# 3. Update your workflows
Follow → MIGRATION.md (Detailed Migration by Category)

# 4. Test and verify
Compare old vs. new results
```

---

## 📚 Documentation Navigation

### Where to Find Information

```
Need to know...                     → Read this document
────────────────────────────────────────────────────────────────
How to get started?                 → docs/QUICKSTART.md
Where did my script go?             → OLD_TO_NEW_SCRIPT_REFERENCE.md
How to migrate workflows?           → MIGRATION.md
How to tune hyperparameters?       → docs/experiments/hyperparameter_tuning.md
How to use cluster?                 → docs/experiments/cluster_guide.md
What changed and why?               → This file (REFACTORING_SUMMARY.md)
All documentation links?            → docs/README.md
Technical details?                  → code.plan.md
```

---

## 🗂️ New Structure Visual Guide

### Before (v1.x) - Flat, Disorganized

```
gunfolds/scripts/
├── FASK_fig4.py              ┐
├── GIMME_fig4.py             │
├── MVAR_fig4.py              │ 6 nearly identical files
├── MVGC_fig4.py              │ (only differ in method name)
├── PCMCI_fig4.py             │
├── RASL_fig4.py              ┘
├── plot_fmri_enhanced.py     ┐
├── plot_gcm_enhanced.py      │ 3 nearly identical files
├── plot_manuscript_compact.py┘
├── lineal_stat_scan.py       ┐
├── lineal_stat_scan2.py      │
├── lineal_stat_scan_config_test.py
├── linear_stat_continious_weights.py
├── linear_stat_continious_weights_n5.py
├── linear_stat_continious_weights_n6.py
├── linear_stat_continious_weights_n7.py   9 similar files!
├── linear_stat_continious_weights_n8.py┘
├── ... (109+ more scripts)
└── (14+ markdown guide files)

Total: 118+ scripts, massive duplication
```

### After (v2.0) - Organized by Function

```
gunfolds/scripts/
├── analysis/                    # Result analysis
│   ├── result_analyzer.py       ← Replaces 5 scripts
│   └── solution_comparer.py
├── benchmarks/                  # Method comparisons
│   ├── benchmark_runner.py      ← Replaces 6 fig4 scripts
│   ├── time_undersampling.py    ← Replaces 8 undersampling scripts
│   └── multi_rasl_runner.py     ← Replaces 6 RASL scripts
├── experiments/                 # VAR experiments
│   ├── var_analyzer.py          ← Replaces 3 VAR analysis scripts
│   ├── var_hyperparameter_tuning.py
│   └── var_collector.py
├── visualization/               # Plotting
│   ├── network_plots.py         ← Replaces 3 enhanced plotting scripts
│   ├── manuscript_figures.py
│   └── plotting_utils.py
├── simulation/                  # Data generation
│   ├── linear_stat_scanner.py   ← Replaces 9 linear stat scripts
│   ├── weighted_solver.py       ← Replaces 3 weighted scripts
│   └── data_generator.py
├── real_data/                   # Real datasets
│   ├── fmri_experiment.py       (moved, not merged)
│   ├── roebroeck_gcm.py
│   └── macaque_analysis.py
├── cluster/                     # SLURM/cluster
│   ├── slurm_templates.py
│   └── job_submission.sh
└── legacy/                      # Archived old code
    ├── archive_old/
    ├── experimental/
    ├── testing/
    └── variants/

Total: ~33 core scripts, no duplication
```

---

## 📊 Consolidation Statistics

### Scripts Merged

| Category | Old Scripts | New Script | Reduction |
|----------|-------------|------------|-----------|
| **Fig4 Benchmarks** | 6 scripts (~350 lines each) | 1 unified script | **-5 files** |
| **Enhanced Plotting** | 3 scripts (530/412/230 lines) | 1 unified script | **-2 files** |
| **Time Undersampling** | 8 scripts (4 local + 4 SLURM) | 1 unified script | **-7 files** |
| **Multiple RASL** | 6 scripts | 1 unified script | **-5 files** |
| **Linear Stat Scanning** | 9 scripts | 1 unified script | **-8 files** |
| **VAR Analysis** | 3 scripts | 1 unified script | **-2 files** |
| **Weighted Solver** | 3 scripts | 1 unified script | **-2 files** |
| **Result Analysis** | 5 scripts | 1 unified script | **-4 files** |
| **Documentation** | 14+ markdown files | 4 main guides | **-10+ files** |
| **Total** | **~118 scripts** | **~33 scripts** | **-85 files** |

### Code Duplication Eliminated

- **Before**: 6 fig4 scripts × 350 lines = 2,100 lines of duplicated code
- **After**: 1 script × 350 lines = 350 lines
- **Reduction**: **1,750 lines of duplicated code removed**

Similar reductions across all categories.

---

## 🔄 Migration Path by User Type

### Casual User (Run scripts occasionally)

**Path**: Quick reference approach

1. ✅ Read [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md)
2. ✅ Find your old script in the A-Z table
3. ✅ Run the new equivalent command
4. ✅ Done!

**Time**: 5-10 minutes

---

### Regular User (Run scripts frequently)

**Path**: Thorough migration

1. ✅ Read [MIGRATION.md](MIGRATION.md) overview
2. ✅ Review "Detailed Migration by Category" for scripts you use
3. ✅ Update bookmarks/aliases/automation scripts
4. ✅ Read [docs/QUICKSTART.md](docs/QUICKSTART.md) for new features
5. ✅ Test workflows on sample data

**Time**: 30-60 minutes

---

### Power User (Custom workflows, automation)

**Path**: Complete migration + optimization

1. ✅ Read [MIGRATION.md](MIGRATION.md) completely
2. ✅ Review [code.plan.md](code.plan.md) for technical details
3. ✅ Update all automation scripts/pipelines
4. ✅ Update import statements in custom Python code
5. ✅ Explore new parameterization options (e.g., `--method all`)
6. ✅ Read [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)
7. ✅ Optimize workflows using new batch capabilities

**Time**: 2-4 hours

---

### Cluster User (SLURM/HPC)

**Path**: Cluster-specific migration

1. ✅ Read [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)
2. ✅ Review [MIGRATION.md](MIGRATION.md) for SLURM script changes
3. ✅ Update SLURM job scripts (add `--slurm` flags)
4. ✅ Test single job before batch submission
5. ✅ Update result collection scripts

**Time**: 1-2 hours

---

## 📖 Key Documentation Features

### MIGRATION.md

**What it includes**:
- ✅ Overview of changes and why
- ✅ Complete old → new script mapping tables
- ✅ Detailed category-by-category migration instructions
- ✅ 10+ command-line examples (before/after)
- ✅ Breaking changes list
- ✅ Import path updates
- ✅ 10 FAQ with answers
- ✅ Migration checklist
- ✅ Timeline for deprecations

**Length**: ~1,500 lines (comprehensive!)

---

### OLD_TO_NEW_SCRIPT_REFERENCE.md

**What it includes**:
- ✅ Complete A-Z alphabetical listing
- ✅ Direct mapping: old script → new script + command
- ✅ Quick lookup tables by category
- ✅ 5 detailed migration examples
- ✅ Parameter name changes
- ✅ Import path changes
- ✅ Verification instructions

**Length**: ~800 lines

**Perfect for**: Quick lookups

---

### docs/QUICKSTART.md

**What it includes**:
- ✅ Installation instructions
- ✅ Basic concepts
- ✅ Common tasks (5 examples)
- ✅ Script organization guide
- ✅ Complete first analysis walkthrough (5 steps)
- ✅ Command-line help guide
- ✅ Configuration file examples
- ✅ Batch processing tips
- ✅ Troubleshooting common issues
- ✅ Quick command reference

**Length**: ~600 lines

**Perfect for**: Getting started quickly

---

### docs/experiments/hyperparameter_tuning.md

**What it includes**:
- ✅ VAR model hyperparameters (order, regularization, thresholds)
- ✅ RASL solver parameters (depth, timeout, optimization method)
- ✅ Systematic tuning workflow (4 steps)
- ✅ Analysis tools and visualization
- ✅ Best practices (6 tips)
- ✅ 3 complete example tuning studies
- ✅ Advanced topics (multi-objective, Bayesian optimization)

**Length**: ~800 lines

**Perfect for**: Optimizing performance

---

### docs/experiments/cluster_guide.md

**What it includes**:
- ✅ SLURM basics (concepts, commands)
- ✅ Submitting jobs (2 methods)
- ✅ SLURM directives explained (table)
- ✅ Monitoring and managing jobs
- ✅ Collecting results
- ✅ Best practices (7 tips)
- ✅ 4 common use cases (complete examples)
- ✅ Troubleshooting (6 common problems + solutions)
- ✅ Advanced topics (dependencies, GPU, interactive)

**Length**: ~700 lines

**Perfect for**: Cluster computing

---

## 🎯 Example Use Cases

### Use Case 1: I Used to Run MVAR_fig4.py

**Old way**:

```bash
python MVAR_fig4.py --nodes 10 --samples 500 --output results/mvar_fig4.png
```

**New way**:

```bash
python benchmarks/benchmark_runner.py \
    --method MVAR \
    --nodes 10 \
    --samples 500 \
    --output results/mvar_fig4.png
```

**Where to find detailed info**:
- Quick lookup: [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) (line 45)
- Detailed migration: [MIGRATION.md](MIGRATION.md) (Section: "Fig4 Benchmark Scripts")
- New features: [docs/QUICKSTART.md](docs/QUICKSTART.md) (Section: "Benchmarks")

---

### Use Case 2: I Have a Script That Imports Old Code

**Old import**:

```python
from gunfolds.scripts.plot_fmri_enhanced import plot_network_circular
```

**New import**:

```python
from gunfolds.scripts.visualization.network_plots import plot_network_circular
```

**Where to find detailed info**:
- Import changes: [MIGRATION.md](MIGRATION.md) (Section: "Breaking Changes" → "Import Paths")
- Full mapping: [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) (Section: "Import Path Changes")

---

### Use Case 3: I Run SLURM Jobs on a Cluster

**Old way**: Separate SLURM scripts

```bash
python slurm_MVAR_time_undersampling_data.py --undersampling 2,4,8
```

**New way**: Same script + `--slurm` flag

```bash
python benchmarks/time_undersampling.py \
    --method MVAR \
    --undersampling 2,4,8 \
    --slurm \
    --submit
```

**Where to find detailed info**:
- Quick lookup: [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md) (Section: "S")
- SLURM guide: [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)
- Examples: [MIGRATION.md](MIGRATION.md) (Section: "Example 2: Time Undersampling Study on Cluster")

---

### Use Case 4: I Need to Tune Hyperparameters

**What you need**:

```bash
# Run systematic tuning
python experiments/var_hyperparameter_tuning.py \
    --config tuning_config.yaml \
    --cv-folds 5 \
    --output-dir results/tuning/

# Analyze tuning results
python experiments/var_analyzer.py \
    --analysis-type hyperparameters \
    --input results/tuning/results.csv \
    --output-dir results/analysis/
```

**Where to find detailed info**:
- Complete guide: [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)
- Quick examples: [docs/QUICKSTART.md](docs/QUICKSTART.md) (Section: "Task 5: Hyperparameter Tuning")

---

## ✅ Migration Checklist

Use this checklist to track your migration progress:

### Essential Tasks

- [ ] Read [MIGRATION.md](MIGRATION.md) overview
- [ ] Identify scripts you use (check automation/cron jobs)
- [ ] Look up new equivalents in [OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md)
- [ ] Test new scripts with `--help` flag
- [ ] Update bookmarks/aliases

### If You Have Automation Scripts

- [ ] Update script paths in automation
- [ ] Update command-line parameters
- [ ] Test workflows on sample data
- [ ] Verify results match old scripts

### If You Have Custom Python Code

- [ ] Update import statements
- [ ] Check for parameter name changes
- [ ] Test custom code with new scripts

### For Cluster Users

- [ ] Read [docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)
- [ ] Update SLURM job scripts
- [ ] Test single job before batch
- [ ] Update result collection scripts

### Optional (Recommended)

- [ ] Read [docs/QUICKSTART.md](docs/QUICKSTART.md) for new features
- [ ] Explore new batch capabilities (`--method all`)
- [ ] Read [docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)
- [ ] Update project documentation/README

---

## 🆘 Getting Help

### Documentation Index

```
Question/Need                           → Documentation File
──────────────────────────────────────────────────────────────────────
Where is my old script?                 → OLD_TO_NEW_SCRIPT_REFERENCE.md
How do I migrate?                       → MIGRATION.md
How do I get started?                   → docs/QUICKSTART.md
How do I tune parameters?               → docs/experiments/hyperparameter_tuning.md
How do I use the cluster?               → docs/experiments/cluster_guide.md
What are all the documentation files?   → docs/README.md
What changed technically?               → code.plan.md
```

### Support Channels

1. **Documentation** - Check relevant docs first (see table above)
2. **Script Help** - Run with `--help` flag
3. **GitHub Issues** - Report bugs or missing functionality
4. **GitHub Discussions** - Ask questions, share workflows

---

## 📅 Timeline & Deprecation

| Date | Event |
|------|-------|
| **Dec 2025** | v2.0.0 released with full refactoring |
| **Dec 2025 - Mar 2026** | Deprecation period (old scripts in `legacy/` work as before) |
| **Mar 2026** | v2.1.0 may remove deprecated wrapper scripts |
| **Jun 2026** | `legacy/` scripts may be archived to separate branch |

**Current status**: ✅ v2.0.0 released - Full backward compatibility via `legacy/` folder

---

## 🎉 Benefits Summary

### For Users

✅ **Easier to find scripts** - Logical folder structure  
✅ **Less confusion** - No more wondering which variant to use  
✅ **Better documentation** - 4 comprehensive guides instead of 14 scattered files  
✅ **More features** - Batch mode (`--method all`), unified configs  
✅ **Backward compatible** - Old scripts still work via `legacy/` folder

### For Developers

✅ **Easier maintenance** - Fix bugs once, not 6 times  
✅ **Faster development** - Shared utilities, no duplication  
✅ **Better testing** - Fewer files to test  
✅ **Cleaner codebase** - Professional appearance  
✅ **Easier onboarding** - New contributors understand structure

### For Projects

✅ **Reproducibility** - Old scripts preserved in `legacy/` for exact reproduction  
✅ **Future-proof** - Parameterized design supports new features  
✅ **Scalability** - Unified scripts support batch operations  
✅ **Professional** - Well-organized, well-documented codebase

---

## 📞 Contact & Support

- **GitHub Repository**: https://github.com/your-org/gunfolds
- **Issues**: https://github.com/your-org/gunfolds/issues
- **Discussions**: https://github.com/your-org/gunfolds/discussions
- **Documentation**: This repository (`/docs` folder)

---

## 🙏 Acknowledgments

This refactoring was planned and documented to minimize disruption while maximizing long-term benefits. Thank you for your patience during the transition!

**Questions?** Start with the [documentation index](#-getting-help) above.

---

**Version**: 2.0.0  
**Last Updated**: December 2, 2025  
**Maintained by**: Gunfolds Development Team

---

## Quick Navigation

- **[MIGRATION.md](MIGRATION.md)** - Complete migration guide
- **[OLD_TO_NEW_SCRIPT_REFERENCE.md](OLD_TO_NEW_SCRIPT_REFERENCE.md)** - Quick script lookup
- **[docs/README.md](docs/README.md)** - Documentation hub
- **[docs/QUICKSTART.md](docs/QUICKSTART.md)** - Getting started
- **[docs/experiments/hyperparameter_tuning.md](docs/experiments/hyperparameter_tuning.md)** - Tuning guide
- **[docs/experiments/cluster_guide.md](docs/experiments/cluster_guide.md)** - Cluster guide
- **[code.plan.md](code.plan.md)** - Technical details

**Happy analyzing!** 🚀

