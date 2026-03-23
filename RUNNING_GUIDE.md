# Gunfolds Running Guide

Minimal guide to get experiments running. For full API docs see the [Online Documentation](https://neuroneural.github.io/gunfolds/).

---

## Table of Contents

- [Installation](#installation)
- [Repository Layout](#repository-layout)
- [Real fMRI Data](#real-fmri-data)
  - [Pipeline Overview](#pipeline-overview)
  - [Submitting the Full Experiment](#submitting-the-full-experiment)
  - [Submitting a Partial Experiment](#submitting-a-partial-experiment)
  - [GCM Baseline Experiment](#gcm-baseline-experiment)
  - [Analysis](#analysis)
- [Simulation Experiments](#simulation-experiments)
- [SLURM Quick Reference](#slurm-quick-reference)

---

## Installation

```bash
# Clone and install in editable mode
git clone https://github.com/neuroneural/gunfolds.git
cd gunfolds
pip install -e .

# Required extras
conda install -c conda-forge graph-tool   # or: brew install graph-tool
pip install clingo tigramite
```

On a SLURM cluster, use your conda environment:

```bash
module load anaconda
conda activate multi_v3   # or your env name
```

---

## Repository Layout

```
gunfolds/
├── gunfolds/                 # Python package
│   ├── solvers/              #   clingo_rasl, clingo_msl, traversal, ...
│   ├── estimation/           #   pc, linear_model, grangercausality, ...
│   ├── utils/                #   bfutils, graphkit, zickle, ...
│   ├── viz/                  #   gtool, dbn2latex
│   └── scripts/
│       ├── cluster/          #   SLURM submit & worker scripts
│       ├── real_data/        #   fmri_experiment_large.py, component_config.py
│       ├── analysis/         #   analyze_fmri_experiment.py
│       ├── simulation/       #   PCMCI, DRASL, gendata sims
│       ├── benchmarks/       #   benchmark_runner.py, time_undersampling.py
│       └── experiments/      #   VAR, hyperparameter tuning
├── tests/
├── setup.py
└── README.md
```

---

## Real fMRI Data

The fMRI pipeline runs RASL (and optionally PCMCI / GCM) causal discovery on FBIRN ICA data (310 subjects, HC vs SZ) via SLURM array jobs.

### Pipeline Overview

```
┌─────────────────────────────────────────────────────┐
│  YOU (on login node)                                │
│  bash submit_fmri_experiment.sh 310                 │
└──────────────────────┬──────────────────────────────┘
                       │
                       │  calls sbatch 6 times (2 configs x 3 partitions)
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐   ┌──────────┐   ┌──────────┐
  │ qTRDGPU  │   │ qTRDHM   │   │ qTRD     │
  │ subj 0-  │   │ subj 104-│   │ subj 207-│
  │ 103      │   │ 206      │   │ 309      │
  │ (N20+N53)│   │ (N20+N53)│   │ (N20+N53)│
  │ 10 ∥ each│   │ 10 ∥ each│   │ 10 ∥ each│
  └────┬─────┘   └────┬─────┘   └────┬─────┘
       │               │              │
       │  Each array task runs:       │
       ▼               ▼              ▼
┌─────────────────────────────────────────────────────┐
│  slurm_fmri_large.sh  (per subject, per config)     │
│  ├── activates conda env (multi_v3)                 │
│  ├── sets OMP_NUM_THREADS=15                        │
│  └── runs:                                          │
│                                                     │
│      python fmri_experiment_large.py                │
│          --subject_idx $SUBJECT_IDX                 │
│          --n_components {20|53}                      │
│          --scc_strategy domain                      │
│          --method RASL                              │
│          --timestamp $TIMESTAMP                     │
│          --PNUM 15 --MAXU 4 --PRIORITY 11112        │
│          --selection_mode top_k --top_k 10          │
└────────────────────┬────────────────────────────────┘
                     │
                     │  Python script imports from:
                     ▼
┌─────────────────────────────────────────────────────┐
│  fmri_experiment_large.py                           │
│  ├── component_config.py                            │
│  │   ├── get_comp_indices(n_components)             │
│  │   ├── get_comp_names(comp_indices)               │
│  │   └── get_scc_members("domain", comp_indices)    │
│  │       └── get_domain_sccs(comp_indices)           │
│  ├── gunfolds (bfutils, conversions, graphkit,      │
│  │             zickle, solvers.clingo_rasl)          │
│  └── loads fMRI data from .mat file                 │
└────────────────────┬────────────────────────────────┘
                     │
                     │  saves output
                     ▼
┌─────────────────────────────────────────────────────┐
│  fbirn_results/<TIMESTAMP>/                         │
│  ├── N20_domain_RASL/                               │
│  │   ├── subject_000/result.zkl                     │
│  │   ├── subject_001/result.zkl                     │
│  │   └── ...                                        │
│  └── N53_domain_RASL/                               │
│      ├── subject_000/result.zkl                     │
│      ├── subject_001/result.zkl                     │
│      └── ...                                        │
└────────────────────┬────────────────────────────────┘
                     │
                     │  after all 620 jobs finish
                     ▼
┌─────────────────────────────────────────────────────┐
│  YOU (manually)                                     │
│  python analyze_fmri_experiment.py                  │
│      --timestamp <TIMESTAMP> --plot                 │
└─────────────────────────────────────────────────────┘
```

**Summary of the chain:**

1. **`submit_fmri_experiment.sh`** — You run this once. It splits 310 subjects into 3 groups and calls `sbatch` 6 times (N=20 + N=53 for each of the 3 partitions).

2. **`slurm_fmri_large.sh`** — SLURM runs this once per subject per config (620 times total). It sets up the environment and launches the Python script.

3. **`fmri_experiment_large.py`** — The actual experiment. It loads fMRI data for one subject, gets domain-based SCCs from `component_config.py`, runs the RASL solver (clingo), and saves `result.zkl`.

4. **`analyze_fmri_experiment.py`** — You run this manually after all jobs complete. It aggregates all 620 result files and produces comparison plots/statistics.


### Submitting the Full Experiment

**Script:** `gunfolds/scripts/cluster/submit_fmri_experiment.sh`

```bash
cd gunfolds/scripts/real_data
bash ../cluster/submit_fmri_experiment.sh [N_SUBJECTS] [GT_DENSITY_MODE] [VALUE]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `N_SUBJECTS` | no | `310` | Total number of subjects to process |
| `GT_DENSITY_MODE` | no | `none` | Density constraint: `none`, `fixed`, or `fraction` |
| `VALUE` | no | `75` (fixed) / `0.5` (fraction) | Density value for the chosen mode |

**Examples:**

```bash
# Default: 310 subjects, no density constraint, N=20 + N=53 domain RASL
bash ../cluster/submit_fmri_experiment.sh 310

# With fixed GT density (value 0–1000, represents density*1000)
bash ../cluster/submit_fmri_experiment.sh 310 fixed 75

# With fractional GT density (fraction of estimated graph density)
bash ../cluster/submit_fmri_experiment.sh 310 fraction 0.5
```

This submits **620 jobs** (310 subjects x 2 configs) split across 3 SLURM partitions (`qTRDGPU`, `qTRDHM`, `qTRD`), with 10 parallel tasks per partition.


### Submitting a Partial Experiment

**Script:** `gunfolds/scripts/cluster/submit_fmri_experiment_partial.sh`

Runs a single config (N=10, domain, RASL) on only `qTRDGPU`. Useful for quick partial runs or when only the GPU partition is available.

```bash
cd gunfolds/scripts/real_data
bash ../cluster/submit_fmri_experiment_partial.sh [N_SUBJECTS] [GT_DENSITY_MODE] [VALUE]
```

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `N_SUBJECTS` | no | `310` | Total number of subjects to process |
| `GT_DENSITY_MODE` | no | `none` | Density constraint: `none`, `fixed`, or `fraction` |
| `VALUE` | no | `75` (fixed) / `0.5` (fraction) | Density value for the chosen mode |

**Examples:**

```bash
bash ../cluster/submit_fmri_experiment_partial.sh 310
bash ../cluster/submit_fmri_experiment_partial.sh 310 fixed 75
bash ../cluster/submit_fmri_experiment_partial.sh 310 fraction 0.5
```


### GCM Baseline Experiment

**Script:** `gunfolds/scripts/cluster/submit_gcm_parallel.sh`

```bash
bash ../cluster/submit_gcm_parallel.sh [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `-A`, `--alpha` | `50` | GCM significance level (50 = 0.05) |
| `-n`, `--num-subjects` | `100` | Number of subjects |
| `-c`, `--concurrent` | `20` | Max concurrent SLURM tasks |
| `-h`, `--help` | — | Show help |


### Analysis

After all jobs finish, aggregate results and generate plots:

```bash
cd gunfolds/scripts/real_data

# Basic analysis
python ../analysis/analyze_fmri_experiment.py --timestamp <TIMESTAMP>

# With plots
python ../analysis/analyze_fmri_experiment.py --timestamp <TIMESTAMP> --plot

# With FDR correction instead of Bonferroni
python ../analysis/analyze_fmri_experiment.py --timestamp <TIMESTAMP> --plot --correction fdr
```

| Flag | Default | Description |
|------|---------|-------------|
| `--timestamp` | *required* | Shared timestamp from the submission step |
| `--results_root` | `fbirn_results` | Root directory containing timestamped results |
| `--data_path` | `../fbirn/fbirn_sz_data.npz` | Path to FBIRN label data |
| `--plot` | off | Generate comparison bar charts and heatmaps |
| `--alpha` | `0.05` | Significance level for edge-level tests |
| `--correction` | `bonferroni` | Multiple-comparison correction: `bonferroni` or `fdr` |

Output goes to `fbirn_results/<TIMESTAMP>/analysis/<correction>/`.


### `fmri_experiment_large.py` — All Flags

You normally don't call this directly (SLURM does), but for local testing:

```bash
python fmri_experiment_large.py --subject_idx 0 --n_components 20 \
    --scc_strategy domain --method RASL --timestamp 03012026120000
```

| Flag | Default | Choices / Range | Description |
|------|---------|-----------------|-------------|
| `--n_components` | `20` | `10`, `20`, `53` | Number of ICA components |
| `--scc_strategy` | `domain` | `domain`, `correlation`, `estimated`, `none` | SCC grouping strategy |
| `--method` | `RASL` | `RASL`, `PCMCI`, `GCM` | Causal discovery method |
| `--subject_idx` | `1` | `0–309` | Subject index (for SLURM array) |
| `--timestamp` | auto | — | Shared timestamp for result grouping |
| `--data_path` | `../fbirn/fbirn_sz_data.npz` | — | Path to FBIRN data |
| `-p`, `--PNUM` | auto | — | Number of CPUs for clingo |
| `-x`, `--MAXU` | `4` | — | Max undersampling rate |
| `-y`, `--PRIORITY` | `11112` | — | Edge weight priorities |
| `--selection_mode` | `top_k` | `top_k`, `delta_threshold` | Solution selection mode |
| `--top_k` | `10` | — | Top k solutions to keep |
| `--delta_multiplier` | `1.9` | — | Delta multiplier for threshold selection |
| `--gt_density_mode` | `none` | `none`, `fixed`, `fraction` | GT density mode |
| `--gt_density` | `75` | `0–1000` | Fixed GT density value |
| `--gt_density_fraction` | `0.5` | `0–1` | Fraction of estimated density |
| `--gcm_alpha` | `0.01` | — | GCM significance level |
| `--gcm_pmax` | `8` | — | GCM max VAR lag order |
| `--gcm_nboot` | `200` | — | GCM bootstrap surrogates |
| `--corr_max_cluster` | `8` | — | Max cluster size (correlation SCC) |

---

## Simulation Experiments

Simulation scripts live under `gunfolds/scripts/simulation/` and `gunfolds/scripts/experiments/`. A typical local run:

```bash
cd gunfolds/scripts/experiments
python VAR_experiment.py
```

Benchmarks:

```bash
cd gunfolds/scripts/benchmarks
python benchmark_runner.py --method RASL
python time_undersampling.py --method RASL
```

---

## SLURM Quick Reference

```bash
# Monitor your jobs
squeue -u $USER

# Cancel a specific job
scancel <JOBID>

# Cancel all your jobs
scancel -u $USER

# Check job history
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed

# Watch output in real time
tail -f out/fmri_out<JOBID>-<TASKID>.out

# Check for errors
grep -i error err/fmri_error*.err
```

For a comprehensive SLURM guide, see [`cluster_guide.md`](cluster_guide.md).
