# Migration Workflow Visual Guide

**Step-by-step visual guide for migrating from old to new codebase**

---

## 🗺️ Your Migration Journey

```
┌─────────────────────────────────────────────────────────────┐
│                     START HERE                              │
│                                                             │
│  Are you familiar with the old codebase?                   │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴──────────┐
                │                      │
            YES │                      │ NO
                │                      │
                ▼                      ▼
    ┌──────────────────────┐  ┌─────────────────────┐
    │  EXISTING USER       │  │   NEW USER          │
    │  PATH                │  │   PATH              │
    └──────────────────────┘  └─────────────────────┘
                │                      │
                │                      ▼
                │          ┌─────────────────────────┐
                │          │ Read:                   │
                │          │ docs/QUICKSTART.md      │
                │          │                         │
                │          │ You're done! 🎉         │
                │          └─────────────────────────┘
                │
                ▼
    ┌──────────────────────────────────────┐
    │ Step 1: Choose Your Migration Style  │
    └──────────────────────────────────────┘
                │
    ┌───────────┴──────────┬──────────────┐
    │                      │              │
    ▼                      ▼              ▼
┌─────────┐      ┌──────────────┐   ┌────────────┐
│ QUICK   │      │ THOROUGH     │   │ COMPLETE   │
│ 5-10min │      │ 30-60min     │   │ 2-4 hours  │
└─────────┘      └──────────────┘   └────────────┘
```

---

## 🚀 Quick Migration (5-10 minutes)

**For**: Casual users who run scripts occasionally

```
┌────────────────────────────────────────────────────────┐
│ Step 1: Open the quick reference                      │
│                                                        │
│ 📄 OLD_TO_NEW_SCRIPT_REFERENCE.md                     │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 2: Find your script (Ctrl+F / Cmd+F)             │
│                                                        │
│ Example: Search "MVAR_fig4"                            │
│                                                        │
│ Found:                                                 │
│   Old: MVAR_fig4.py                                    │
│   New: benchmarks/benchmark_runner.py --method MVAR    │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 3: Run the new command                            │
│                                                        │
│ $ python benchmarks/benchmark_runner.py \              │
│       --method MVAR \                                  │
│       --nodes 10 \                                     │
│       --samples 500                                    │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                   ✅ DONE!                             │
│                                                        │
│ Optional: Bookmark the reference file for next time    │
└────────────────────────────────────────────────────────┘
```

### Quick Reference Table

| Old Script | New Command |
|------------|-------------|
| `MVAR_fig4.py` | `benchmarks/benchmark_runner.py --method MVAR` |
| `plot_fmri_enhanced.py` | `visualization/network_plots.py --data-source fmri` |
| `VAR_analyze_delta.py` | `experiments/var_analyzer.py --analysis-type delta` |

---

## 📚 Thorough Migration (30-60 minutes)

**For**: Regular users who want to understand the new structure

```
┌────────────────────────────────────────────────────────┐
│ Step 1: Read MIGRATION.md overview (10 min)           │
│                                                        │
│ 📄 MIGRATION.md (sections: Overview, Quick Reference) │
│                                                        │
│ Learn:                                                 │
│ • Why the refactoring                                  │
│ • New directory structure                              │
│ • High-level changes                                   │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 2: Identify your commonly-used scripts (5 min)   │
│                                                        │
│ Review your:                                           │
│ • Shell history: $ history | grep "python"             │
│ • Bookmarks/aliases                                    │
│ • Automation scripts/cron jobs                         │
│                                                        │
│ List: ________________________________                 │
│       ________________________________                 │
│       ________________________________                 │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 3: Read detailed migration for your scripts      │
│         (15-20 min)                                    │
│                                                        │
│ 📄 MIGRATION.md (section: Detailed Migration)         │
│                                                        │
│ For each script you use:                               │
│ • Read "Old Workflow" example                          │
│ • Read "New Workflow" example                          │
│ • Read "Key Changes"                                   │
│ • Note any breaking changes                            │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 4: Update your workflows (15-20 min)             │
│                                                        │
│ Update:                                                │
│ ☐ Shell aliases                                        │
│ ☐ Bookmarks                                            │
│ ☐ Automation scripts                                   │
│ ☐ Documentation/notes                                  │
│                                                        │
│ Test new commands on sample data                       │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 5: Read QUICKSTART.md (10-15 min)                │
│                                                        │
│ 📄 docs/QUICKSTART.md                                  │
│                                                        │
│ Discover new features:                                 │
│ • Batch operations (--method all)                      │
│ • Unified configuration files                          │
│ • Better parameterization                              │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                   ✅ DONE!                             │
│                                                        │
│ You're now familiar with the new structure and can     │
│ take advantage of new features!                        │
└────────────────────────────────────────────────────────┘
```

---

## 🔧 Complete Migration (2-4 hours)

**For**: Power users with custom workflows, automation, and imports

```
┌────────────────────────────────────────────────────────┐
│ Phase 1: Understanding (30-45 min)                     │
└────────────────────────────────────────────────────────┘
    │
    ├─► Read MIGRATION.md completely (20 min)
    │   • Overview & rationale
    │   • All category-by-category migrations
    │   • Breaking changes
    │   • FAQ
    │
    ├─► Read code.plan.md (15 min)
    │   • Technical details
    │   • Implementation approach
    │   • Expected benefits
    │
    └─► Review REFACTORING_SUMMARY.md (10 min)
        • Consolidation statistics
        • Visual structure comparison
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Phase 2: Inventory (20-30 min)                        │
└────────────────────────────────────────────────────────┘
    │
    ├─► Identify all scripts you use
    │   $ grep -r "python.*gunfolds" ~/scripts/
    │   $ crontab -l | grep gunfolds
    │
    ├─► List custom Python imports
    │   $ grep -r "from gunfolds.scripts" ~/code/
    │
    ├─► Document current workflows
    │   Create migration_notes.md with:
    │   • Scripts used
    │   • Frequency of use
    │   • Dependencies
    │
    └─► Identify automation scripts
        • Shell scripts
        • Cron jobs
        • CI/CD pipelines
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Phase 3: Update Code (45-60 min)                      │
└────────────────────────────────────────────────────────┘
    │
    ├─► Update automation scripts (20 min)
    │   For each automation script:
    │   1. Back up original: cp script.sh script.sh.backup
    │   2. Update script paths
    │   3. Update parameters
    │   4. Add comments for changes
    │
    ├─► Update import statements (15 min)
    │   Old: from gunfolds.scripts.plot_fmri_enhanced import X
    │   New: from gunfolds.scripts.visualization.network_plots import X
    │
    ├─► Update shell aliases (10 min)
    │   Edit ~/.bashrc or ~/.zshrc
    │   Update gunfolds-related aliases
    │
    └─► Update documentation (15 min)
        • Project README files
        • Lab wiki pages
        • Analysis notebooks
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Phase 4: Testing (30-45 min)                          │
└────────────────────────────────────────────────────────┘
    │
    ├─► Create test directory
    │   $ mkdir migration_tests
    │   $ cd migration_tests
    │
    ├─► Test each updated script (20-30 min)
    │   For critical scripts:
    │   1. Run old script (from legacy/)
    │   2. Run new script
    │   3. Compare outputs:
    │      $ diff results_old/ results_new/
    │   4. Verify metrics match
    │
    ├─► Test automation scripts (10 min)
    │   • Run in dry-run mode if available
    │   • Start with subset of data
    │   • Monitor for errors
    │
    └─► Validate custom code (5 min)
        • Run unit tests
        • Check imports work
        │
        ▼
┌────────────────────────────────────────────────────────┐
│ Phase 5: Optimization (30-45 min)                     │
└────────────────────────────────────────────────────────┘
    │
    ├─► Explore new features (15 min)
    │   • Batch mode: --method all
    │   • Config files: --config config.yaml
    │   • Better parallelization
    │
    ├─► Read hyperparameter tuning guide (15 min)
    │   📄 docs/experiments/hyperparameter_tuning.md
    │
    ├─► Optimize workflows (10 min)
    │   Old: Run 6 scripts sequentially
    │   New: python benchmark_runner.py --method all
    │
    └─► Update cluster scripts if applicable (15 min)
        📄 docs/experiments/cluster_guide.md
        │
        ▼
┌────────────────────────────────────────────────────────┐
│                   ✅ DONE!                             │
│                                                        │
│ Full migration complete with optimizations!            │
│                                                        │
│ Next: Share your experience with the team!             │
└────────────────────────────────────────────────────────┘
```

---

## 🖥️ Cluster User Migration (1-2 hours)

**For**: Users running jobs on SLURM/HPC clusters

```
┌────────────────────────────────────────────────────────┐
│ Step 1: Read cluster guide (30 min)                   │
│                                                        │
│ 📄 docs/experiments/cluster_guide.md                   │
│                                                        │
│ Focus on:                                              │
│ • SLURM basics review                                  │
│ • New --slurm flag usage                               │
│ • Job submission changes                               │
│ • Best practices                                       │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 2: Review your SLURM scripts (15 min)            │
│                                                        │
│ Identify:                                              │
│ • Job submission scripts (*.sh)                        │
│ • Python scripts that generate SLURM scripts           │
│ • Cron jobs that submit SLURM jobs                     │
│                                                        │
│ List: ________________________________                 │
│       ________________________________                 │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 3: Update SLURM scripts (30 min)                 │
│                                                        │
│ Key changes:                                           │
│                                                        │
│ Old approach:                                          │
│   • Separate slurm_*.py scripts                        │
│   • Manual job script creation                         │
│                                                        │
│ New approach:                                          │
│   • Add --slurm --submit flags                         │
│   • Automatic job script generation                    │
│                                                        │
│ Example:                                               │
│ OLD: python slurm_MVAR_time_undersampling_data.py      │
│ NEW: python benchmarks/time_undersampling.py \         │
│          --method MVAR --slurm --submit                │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 4: Test with small job (15 min)                  │
│                                                        │
│ $ python benchmarks/benchmark_runner.py \              │
│       --method RASL \                                  │
│       --nodes 5 \                                      │
│       --samples 100 \                                  │
│       --slurm \                                        │
│       --submit \                                       │
│       --time 00:10:00                                  │
│                                                        │
│ Monitor:                                               │
│ $ squeue -u $USER                                      │
│ $ tail -f logs/*.out                                   │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 5: Update result collection (15 min)             │
│                                                        │
│ Old: Manual result collection scripts                  │
│                                                        │
│ New: Use unified analyzer                              │
│ $ python analysis/result_analyzer.py \                 │
│       --data-type simulation \                         │
│       --results-dir $SCRATCH/results/ \                │
│       --output summary.csv                             │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│ Step 6: Scale up (15 min)                             │
│                                                        │
│ Now submit full-scale experiments:                     │
│                                                        │
│ $ python benchmarks/time_undersampling.py \            │
│       --method all \                                   │
│       --nodes 5,10,15,20 \                             │
│       --samples 1000 \                                 │
│       --undersampling 2,4,8 \                          │
│       --slurm \                                        │
│       --submit \                                       │
│       --partition normal \                             │
│       --time 02:00:00 \                                │
│       --mem 16G                                        │
└────────────────────────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────┐
│                   ✅ DONE!                             │
│                                                        │
│ Your cluster workflows are now updated!                │
└────────────────────────────────────────────────────────┘
```

---

## 🎯 Decision Tree: Which Document Do I Need?

```
┌──────────────────────────────────────────────┐
│ What do you need to know?                   │
└──────────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┬─────────────┬──────────────┐
        │                       │             │              │
        ▼                       ▼             ▼              ▼
  ┌──────────┐          ┌──────────┐   ┌──────────┐  ┌──────────┐
  │ Where is │          │   How do │   │   How do │  │   How to │
  │ my old   │          │    I use │   │  I tune  │  │   use    │
  │ script?  │          │  new     │   │  hyper-  │  │  cluster?│
  └──────────┘          │  code?   │   │  params? │  └──────────┘
        │               └──────────┘   └──────────┘        │
        │                     │              │             │
        ▼                     ▼              ▼             ▼
  OLD_TO_NEW_        QUICKSTART.md    hyperparameter_  cluster_
  SCRIPT_                              tuning.md        guide.md
  REFERENCE.md
        │                     │              │             │
        │                     │              │             │
        ▼                     ▼              ▼             ▼
  Quick lookup        Basic usage    Tuning guide    SLURM usage
  table               examples       + examples      + examples
```

---

## 📋 Verification Checklist

After migration, verify your setup:

### ✅ Basic Verification

```bash
# Test 1: Script exists and has help
python benchmarks/benchmark_runner.py --help
# ✓ Should show help message

# Test 2: Can import from new locations
python -c "from gunfolds.scripts.visualization.network_plots import plot_network_circular"
# ✓ Should complete without error

# Test 3: Run simple command
python benchmarks/benchmark_runner.py --method RASL --nodes 5 --samples 10
# ✓ Should run and produce output
```

### ✅ Results Verification

```bash
# Run old script (from legacy)
python scripts/legacy/experimental/MVAR_fig4.py \
    --nodes 10 --samples 100 \
    --output results_old.csv

# Run new script
python benchmarks/benchmark_runner.py \
    --method MVAR --nodes 10 --samples 100 \
    --output results_new.csv

# Compare outputs
python utils/compare_results.py \
    --old results_old.csv \
    --new results_new.csv \
    --tolerance 1e-6
# ✓ Results should match (within numerical tolerance)
```

### ✅ Automation Verification

```bash
# Test automation script (dry run if available)
bash my_automation_script.sh --dry-run
# ✓ Should show commands without executing

# Or run on small test data
bash my_automation_script.sh --test-mode
# ✓ Should complete successfully
```

### ✅ Import Verification

```python
# test_imports.py
try:
    from gunfolds.scripts.visualization.network_plots import plot_network_circular
    from gunfolds.scripts.benchmarks.benchmark_runner import create_boxplot
    from gunfolds.scripts.experiments.var_analyzer import compute_delta_metrics
    print("✓ All imports successful!")
except ImportError as e:
    print(f"✗ Import failed: {e}")
```

---

## 🚦 Migration Status Tracker

Track your migration progress:

```
┌─────────────────────────────────────────────────┐
│ MIGRATION STATUS                                │
├─────────────────────────────────────────────────┤
│                                                 │
│ Phase 1: Documentation                          │
│ ☐ Read MIGRATION.md                             │
│ ☐ Read OLD_TO_NEW_SCRIPT_REFERENCE.md           │
│ ☐ Read QUICKSTART.md                            │
│ ☐ Read relevant advanced guides                 │
│                                                 │
│ Phase 2: Inventory                              │
│ ☐ List all scripts I use                        │
│ ☐ List automation scripts                       │
│ ☐ List custom Python code with imports          │
│ ☐ List SLURM scripts (if applicable)            │
│                                                 │
│ Phase 3: Updates                                │
│ ☐ Update script paths                           │
│ ☐ Update parameters                             │
│ ☐ Update import statements                      │
│ ☐ Update shell aliases/bookmarks                │
│ ☐ Update automation scripts                     │
│ ☐ Update SLURM scripts                          │
│                                                 │
│ Phase 4: Testing                                │
│ ☐ Test critical scripts                         │
│ ☐ Compare old vs new results                    │
│ ☐ Test automation workflows                     │
│ ☐ Test custom code                              │
│                                                 │
│ Phase 5: Documentation                          │
│ ☐ Update project README                         │
│ ☐ Update team wiki/docs                         │
│ ☐ Document changes made                         │
│                                                 │
│ Phase 6: Cleanup                                │
│ ☐ Remove old script bookmarks                   │
│ ☐ Archive old automation scripts                │
│ ☐ Update .gitignore if needed                   │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 💡 Pro Tips

### Tip 1: Use aliases for frequently used commands

```bash
# Add to ~/.bashrc or ~/.zshrc
alias gf-bench='python ~/gunfolds/scripts/benchmarks/benchmark_runner.py'
alias gf-viz='python ~/gunfolds/scripts/visualization/network_plots.py'
alias gf-analyze='python ~/gunfolds/scripts/analysis/result_analyzer.py'

# Usage
gf-bench --method RASL --nodes 10
gf-viz --data-source fmri --output fig.pdf
```

### Tip 2: Create a config file for your common settings

```yaml
# ~/.gunfolds_config.yaml
default_nodes: 10
default_samples: 500
default_output_dir: ~/results/gunfolds/
default_undersampling: 2

preferred_methods:
  - RASL
  - MVAR
  - MVGC
```

### Tip 3: Keep a migration notes file

```markdown
# my_gunfolds_migration.md

## Scripts I migrated
- [x] MVAR_fig4.py → benchmarks/benchmark_runner.py --method MVAR
- [x] plot_fmri_enhanced.py → visualization/network_plots.py --data-source fmri
- [ ] (more scripts...)

## Issues encountered
- None so far!

## New features I'm using
- Batch mode: --method all
- Config files: --config my_config.yaml
```

---

## 🆘 Troubleshooting Decision Tree

```
┌─────────────────────────────────────┐
│ Having issues?                      │
└─────────────────────────────────────┘
            │
    ┌───────┴──────┬──────────┬───────────┐
    │              │          │           │
    ▼              ▼          ▼           ▼
Can't find    Import      Results    Cluster
script?       error?      differ?    issues?
    │              │          │           │
    ▼              ▼          ▼           ▼
OLD_TO_NEW   MIGRATION   MIGRATION   cluster_
SCRIPT_      .md         .md         guide.md
REFERENCE    "Import     "Verifi-    "Trouble-
.md          Paths"      cation"     shooting"
```

---

**Need help?** See documentation files listed above or open a GitHub issue.

**Version**: 2.0.0  
**Last Updated**: December 2, 2025  
**Maintainer**: Gunfolds Development Team

