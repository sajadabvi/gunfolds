# Legacy Scripts Directory

This directory contains **85+ deprecated, experimental, or superseded scripts** that have been archived for reference.

---

## ⚠️ Deprecation Notice

**These scripts are deprecated and may be removed in a future release.**

Please migrate to the new unified scripts. See [`../MIGRATION.md`](../MIGRATION.md) for complete migration instructions.

---

## Why These Scripts Are Here

These scripts have been replaced by unified, more maintainable versions. They are kept here for:

- **Historical reference** - Compare old vs new implementations
- **Backward compatibility** - Transition period support
- **Verification** - Ensure new scripts match old behavior

---

## Quick Migration Reference

### Plotting Scripts → `visualization/network_plots.py`

| Old Script | New Command |
|------------|-------------|
| `plot_fmri_enhanced.py -t TIMESTAMP` | `../visualization/network_plots.py --source fmri --timestamp TIMESTAMP` |
| `plot_gcm_enhanced.py -t TIMESTAMP` | `../visualization/network_plots.py --source gcm --timestamp TIMESTAMP` |
| `plot_manuscript_compact.py` | `../visualization/network_plots.py --compact` |

### Benchmark Scripts → `benchmarks/benchmark_runner.py`

| Old Script | New Command |
|------------|-------------|
| `FASK_fig4.py` | `../benchmarks/benchmark_runner.py --method FASK` |
| `GIMME_fig4.py` | `../benchmarks/benchmark_runner.py --method GIMME` |
| `MVAR_fig4.py` | `../benchmarks/benchmark_runner.py --method MVAR` |
| `MVGC_fig4.py` | `../benchmarks/benchmark_runner.py --method MVGC` |
| `PCMCI_fig4.py` | `../benchmarks/benchmark_runner.py --method PCMCI` |
| `RASL_fig4.py` | `../benchmarks/benchmark_runner.py --method RASL` |

### Time Undersampling Scripts → `benchmarks/time_undersampling.py`

| Old Script | New Command |
|------------|-------------|
| `GIMME_time_undersampling_data.py` | `../benchmarks/time_undersampling.py --method GIMME` |
| `MVAR_time_undersampling_data.py` | `../benchmarks/time_undersampling.py --method MVAR` |
| `MVGC_time_undersampling_data.py` | `../benchmarks/time_undersampling.py --method MVGC` |
| `slurm_*_time_undersampling_data.py` | `../benchmarks/time_undersampling.py --method [METHOD] --slurm` |

### VAR Analysis Scripts → `experiments/var_analyzer.py`

| Old Script | New Command |
|------------|-------------|
| `VAR_analyze_delta.py -f FILE` | `../experiments/var_analyzer.py --analysis-type delta -f FILE` |
| `VAR_analyze_hyperparameters.py -f FILE` | `../experiments/var_analyzer.py --analysis-type hyperparameters -f FILE` |
| `VAR_analyze_orientation_only.py -f FILE` | `../experiments/var_analyzer.py --analysis-type orientation -f FILE` |

### Linear Statistics Scripts → Archived

All `lineal_stat*.py` and `linear_stat*.py` variants archived. Unified version coming soon.

### Weighted Solver Scripts → Archived

All `weighted*.py` variants archived. Unified version coming soon.

### Multiple RASL Scripts → Archived

All `*multiple_rasl.py`, `*multi_indiv*.py` variants archived. Unified version coming soon.

---

## Contents of This Directory

### Archived by Category:

**Benchmark Scripts (6 files):**
- `FASK_fig4.py`
- `GIMME_fig4.py`
- `MVAR_fig4.py`
- `MVGC_fig4.py`
- `PCMCI_fig4.py`
- `RASL_fig4.py`

**Plotting Scripts (3 files):**
- `plot_fmri_enhanced.py`
- `plot_gcm_enhanced.py`
- `plot_manuscript_compact.py`

**Time Undersampling (7 files):**
- `*_time_undersampling_data.py` (4 files)
- `slurm_*_time_undersampling_data.py` (3 files)

**VAR Analysis (3 files):**
- `VAR_analyze_delta.py`
- `VAR_analyze_hyperparameters.py`
- `VAR_analyze_orientation_only.py`

**Linear Statistics (9 files):**
- `lineal_stat_scan.py`
- `lineal_stat_scan2.py`
- `lineal_stat_scan_config_test.py`
- `lineal_stat_scan_dataset.py`
- `linear_stat_continious_weights*.py` (5 variants)

**Weighted Solvers (3 files):**
- `weighted.py`
- `weighted_optN.py`
- `weighted_then_drasl.py`

**RASL Variants (6 files):**
- `GIMME_multiple_rasl.py`
- `MVAR_multi_individual_rasl.py`
- `MVAR_multiple_rasl.py`
- `MVGC_multi_indiv_rasl.py`
- `MVGC_mult_comb_rasl.py`
- `MVGC_multiple_rasl.py`

**Other Variants:**
- `MVGC_expo_impo.py`
- `MVGC_F1_improvment_undersampling.py`
- `MVGC_then_srasl.py`
- `MVGC_undersampeld_GT.py`
- `PCMCI_using_tigramite_VAR.py`

**Test & Experimental:**
- `test_*.py` (2 files)
- `*_retry.py` (1 file)
- Old directory contents

**Documentation (14+ files):**
- All `.md` and `.txt` guide files
- Manuscript files (`.tex`, `.pdf`)
- Figure files (`.png`, `.svg`)

---

## Using Legacy Scripts

### ⚠️ Not Recommended

Legacy scripts should only be used for:
1. **Reference** - Comparing with new implementation
2. **Verification** - Ensuring new scripts match old behavior
3. **Temporary** - During transition period

### If You Must Use Them

```bash
cd legacy/

# Old plotting
python plot_gcm_enhanced.py -t TIMESTAMP

# Old benchmark
python MVGC_fig4.py

# etc.
```

**Warning:** These scripts:
- Are not maintained
- May have bugs
- May break with library updates
- Will eventually be removed

---

## Migration Timeline

- **Now:** Legacy scripts available for reference
- **1-3 months:** Deprecation warnings added
- **3-6 months:** Legacy scripts may be removed

**Action Required:** Migrate to new unified scripts as soon as possible.

---

## Getting Help

1. **Migration Guide:** [`../MIGRATION.md`](../MIGRATION.md) - Complete old→new mappings
2. **Directory Overview:** [`../README.md`](../README.md) - New structure guide
3. **Quick Start:** [`../../docs/QUICKSTART.md`](../../docs/QUICKSTART.md) - Usage examples
4. **Detailed Guides:** [`../../docs/experiments/`](../../docs/experiments/) - Comprehensive documentation

---

## For Maintainers

### When to Remove Legacy Scripts

After verifying:
- [ ] All users migrated to new scripts
- [ ] New scripts produce equivalent results
- [ ] No outstanding issues referencing legacy code
- [ ] Transition period complete (3-6 months)

### How to Remove

```bash
# Archive for permanent reference
cd /path/to/gunfolds/scripts
tar -czf legacy_archive_$(date +%Y%m%d).tar.gz legacy/

# Then remove
rm -rf legacy/
```

---

**Last Updated:** December 2, 2024
**Status:** 85+ scripts archived
**Migration Status:** In progress - users transitioning to new unified scripts

