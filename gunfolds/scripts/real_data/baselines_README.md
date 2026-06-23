# Baseline causal-discovery methods on FBIRN (GIMME / MVAR / MVGC / FASK)

These run four baseline methods on the FBIRN ICA data for **N=10 and N=13**
components and record **one binary directed graph per subject** in the *exact
same* `result.zkl` payload that `refactored_fmri_experiment_large.py` writes for
its single-solution PCMCI / GCM runs. Because the payload is identical, the
existing analysis (`analysis/refactored_analyze_fmri_experiment.py`) picks them
up automatically next to RASL and PCMCI — just share a `--results_root` and
`--timestamp`.

| Method | Backend                         | Execution model                         |
|--------|---------------------------------|-----------------------------------------|
| FASK   | py-tetrad (jpype + tetrad jar)  | per-subject (SLURM array), pure Python  |
| MVGC   | MATLAB (MVGC toolbox)           | one job/N: export → MATLAB → collect    |
| MVAR   | MATLAB (self-contained OLS)     | one job/N: export → MATLAB → collect    |
| GIMME  | R (`gimme` package), pooled     | one job/N: export → Rscript → collect   |

All outputs land in `<results_root>/<TIMESTAMP>/N<N>_none_<METHOD>/subject_<idx>/result.zkl`
(`none` = SCC strategy, which is N/A for these methods). Default
`results_root = fbirn_results_refactored`.

> **Direction invariant.** Every method emits a `[source, target]` adjacency
> (`A[s,t]=1 ⇔ s→t`), matching `cv.graph2adj` and the analysis heat-maps. Each
> external-tool convention is converted explicitly in code (see the module
> docstring of `baselines_fmri_experiment.py`). These baselines do **not** use
> the PCMCI→Glag2CG→drasl path, so the gunfolds checklist items about
> `selfloop`, Glag2CG transposition, density encoding, DD/BD and `urate` do not
> apply.

## Files

```
real_data/baselines_fmri_experiment.py   # driver: FASK run + export/collect for the rest
real_data/baselines_mvgc.m               # MATLAB MVGC (needs the MVGC toolbox)
real_data/baselines_mvar.m               # MATLAB MVAR (self-contained OLS + Wald test)
real_data/baselines_run_gimme.R          # R gimme (pooled group-level)
cluster/slurm_baselines_fask.sh          # FASK array job (one subject/task)
cluster/slurm_baselines_matlab.sh        # MVGC/MVAR pipeline (one job per N+method)
cluster/slurm_baselines_gimme.sh         # GIMME pipeline (one job per N)
cluster/submit_baselines_fmri_experiment.sh  # submit everything for N=10,13
```

## Quick start (cluster)

Run `sbatch`/`bash` from `gunfolds/scripts/real_data/` (same cwd as the
RASL/PCMCI runs, so `../fbirn/fbirn_sz_data.npz` resolves and the `.m`/`.R`
helpers are found). Submit everything:

```bash
cd gunfolds/scripts/real_data
bash ../cluster/submit_baselines_fmri_experiment.sh        # fresh timestamp
# …or compare against an existing RASL/PCMCI run:
TIMESTAMP=06042026120000 bash ../cluster/submit_baselines_fmri_experiment.sh
```

Subset of methods / sizes via env vars:

```bash
METHODS="FASK MVAR" NCOMPS="13" bash ../cluster/submit_baselines_fmri_experiment.sh
```

### Per-method environment knobs

* **FASK** — `TETRAD_JAR` (default `resources/tetrad-current.jar`),
  `PYTETRAD_PATH` (default `~/tread/py-tetrad`), `FASK_ALPHA` (0.05).
* **MVGC/MVAR** — `MVGC_TOOLBOX` (root of the MVGC toolbox, default `~/MVGC`),
  `MATLAB_BIN`/`MATLAB_MODULE`, `ALPHA` (0.05), `MOMAX` (5, MVGC model-order
  search bound), `P` (1, MVAR lag order — FBIRN has only T=140).
* **GIMME** — `R_MODULE` (default `R`), `GIMME_AR` (TRUE), `GROUPCUTOFF` (0.75),
  `SUBCUTOFF` (0.50; subgrouping is OFF).
* **all** — `WORKDIR` (scratch for `.mat`/CSV I/O), `RESULTS_ROOT`,
  `MAX_PARALLEL` (FASK array throttle).

## Manual / local single steps

```bash
# FASK, one subject
python baselines_fmri_experiment.py --method FASK --stage run \
    --subject_idx 42 --n_components 13 --timestamp <TS>

# MVGC: export -> MATLAB -> collect
python baselines_fmri_experiment.py --method MVGC --stage export \
    --n_components 10 --workdir baselines_work/<TS> --timestamp <TS>
matlab -nodisplay -r "addpath(genpath('~/MVGC'));startup; baselines_mvgc('baselines_work/<TS>',10); exit"
python baselines_fmri_experiment.py --method MVGC --stage collect \
    --n_components 10 --workdir baselines_work/<TS> --timestamp <TS>

# GIMME: export -> R -> collect
python baselines_fmri_experiment.py --method GIMME --stage export \
    --n_components 10 --workdir baselines_work/<TS> --timestamp <TS>
Rscript baselines_run_gimme.R baselines_work/<TS>/N10/gimme_in baselines_work/<TS>/N10/gimme_out
python baselines_fmri_experiment.py --method GIMME --stage collect \
    --n_components 10 --workdir baselines_work/<TS> --timestamp <TS>
```

## Analyse alongside RASL / PCMCI

```bash
python ../analysis/refactored_analyze_fmri_experiment.py \
    --timestamp <TS> --results_root fbirn_results_refactored \
    --correction fdr --plot
```

## Dependency notes / caveats

* **FASK** needs jpype + the tetrad jar + py-tetrad on the cluster (the legacy
  FASK pipeline used exactly this). The driver tries `import tools.TetradSearch`
  then `from pytetrad.tools import TetradSearch`. On any failure for a subject
  it logs a warning and saves an empty graph (so the cohort stays complete).
* **MVGC** targets the classic **MVGC v1.0** toolbox API (`tsdata_to_var` →
  `var_to_autocov` → `autocov_to_pwcgc` → `mvgc_pval` → `significance`). On
  MVGC2 swap that block for the v2 equivalents — the I/O contract (read
  `input.mat`, write `sig_<s>.mat` with `A_src_tgt` in `[source,target]`) is
  unchanged, so Python's collect step is untouched.
* **MVAR** is fully self-contained MATLAB (OLS VAR + per-source Wald χ² over the
  block of lag coefficients) — no toolbox required.
* **GIMME** is genuinely group-level; it's run **once over all 311 subjects**
  (pooled). The collect step parses gimme's `indivPathEstimates.csv` (long
  format `file, lhs, op, rhs, …`); a path `lhs ~ rhs` becomes a directed edge
  `rhs → lhs` (AR self-paths dropped). If your gimme version names that file
  differently, adjust `stage_collect_gimme`.
