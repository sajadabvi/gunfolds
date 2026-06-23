"""
baselines_fmri_experiment.py
============================

Baseline causal-discovery methods (GIMME, MVAR, MVGC, FASK) on the FBIRN ICA
data, recording **one binary directed graph per subject** in the *exact same*
``result.zkl`` payload that ``refactored_fmri_experiment_large.py`` writes for
its single-solution PCMCI / GCM runs.  Because the payload is identical, the
existing cross-config analysis
(``analysis/refactored_analyze_fmri_experiment.py``) picks these methods up
automatically alongside RASL and PCMCI -- just point it at the same
``--results_root`` / ``--timestamp``.

Each method produces a single solution (``cost=0``, ``undersampling=None``,
posterior == the binary adjacency), so they slot in next to PCMCI/GCM as
non-undersampling-aware baselines.

----------------------------------------------------------------------------
EXECUTION MODELS DIFFER BY METHOD (this is the whole reason for the --stage flag)
----------------------------------------------------------------------------
  FASK   pure Python, per subject (py-tetrad / jpype + tetrad jar):
             python baselines_fmri_experiment.py --method FASK --stage run \
                 --subject_idx S --n_components 10 --timestamp <TS>

  MVGC /  MATLAB bridge (the toolboxes are MATLAB).  Three steps:
  MVAR     1) python ... --method MVGC --stage export   (writes per-N input.mat)
           2) matlab -r "baselines_mvgc('<workdir>', 10)"  (writes sig_<s>.mat)
           3) python ... --method MVGC --stage collect    (sig -> result.zkl)
         (the slurm script chains all three; MVAR is self-contained MATLAB OLS,
          MVGC needs the MVGC toolbox on the MATLAB path.)

  GIMME  R bridge, GROUP-LEVEL / pooled (the gimme package estimates a shared
         group structure then per-subject deviations).  Three steps:
           1) python ... --method GIMME --stage export   (per-subject CSVs)
           2) Rscript baselines_run_gimme.R <in_dir> <out_dir>
           3) python ... --method GIMME --stage collect   (indivPathEstimates.csv
                                                            -> result.zkl)

----------------------------------------------------------------------------
EDGE-DIRECTION CONVENTION (the one correctness invariant that matters here)
----------------------------------------------------------------------------
Everything downstream (``cv.graph2adj`` and the analysis heat-maps) uses
``A[s, t] == 1  <=>  directed edge  source s -> target t`` (row = source,
col = target).  Every external tool below is converted *into* that convention
explicitly and the conversion is documented at the call site:
  * FASK  : tetrad prints "A --> B" -> A[idx(A), idx(B)] = 1            (already s->t)
  * MVGC  : MVGC's pwcgc F(i,j) = "GC from j to i" ([to, from]); MATLAB saves
            ``A_src_tgt = sig.'`` so Python reads s->t directly.
  * MVAR  : VAR coeff A(i,j,:) = effect of j on i ([to, from]); MATLAB saves
            ``A_src_tgt = sig.'``.
  * GIMME : path "lhs ~ rhs" = outcome ~ predictor -> edge predictor -> outcome,
            i.e. A[idx(rhs_base), idx(lhs)] = 1.

CHECKLIST NOTE (gunfolds script checklist): these baselines do **not** use the
PCMCI -> Glag2CG -> drasl path, so checklist items 1-9 and 11-17 (selfloop=None,
Glag2CG no-transpose, density encoding, DD/BD, urate bound, drasl parsing, BOLD
end_time, randomDAG) DO NOT APPLY -- there is no clingo solve and no Glag2CG
call.  The single cross-cutting concern that *does* apply, edge direction, is
handled explicitly above.

Usage examples:
  # FASK, one subject (SLURM array task)
  python baselines_fmri_experiment.py --method FASK --stage run \
      --subject_idx 42 --n_components 13 --timestamp 06232026120000

  # MVGC export -> (MATLAB) -> collect
  python baselines_fmri_experiment.py --method MVGC --stage export \
      --n_components 10 --workdir baselines_work/06232026120000 --timestamp 06232026120000
  python baselines_fmri_experiment.py --method MVGC --stage collect \
      --n_components 10 --workdir baselines_work/06232026120000 --timestamp 06232026120000
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np

# Make the repo importable when invoked as a script from real_data/ (local
# debugging); on the cluster gunfolds is conda-installed so this is a no-op.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", ".."))

from gunfolds.utils import zickle as zkl
from gunfolds.scripts.real_data.component_config import (
    get_comp_indices, get_comp_names, INDEX_TO_DOMAIN,
)

METHODS = ["FASK", "MVGC", "MVAR", "GIMME"]


# ---------------------------------------------------------------------------
# Shared payload helpers (kept dependency-light: only numpy + zickle + config,
# no tigramite / clingo / igraph import, so FASK/GIMME/MVAR collect run fast)
# ---------------------------------------------------------------------------

def get_labels(npz):
    if "labels" in npz.files:
        return npz["labels"]
    if "label" in npz.files:
        return npz["label"]
    raise KeyError("Labels not found in NPZ. Expected 'labels' or 'label'.")


def config_tag(n_components, method):
    """Mirror the RASL/PCMCI tag shape (N<k>_<scc>_<method>); SCC is N/A here."""
    return f"N{n_components}_none_{method}"


def subject_dir(results_root, timestamp, n_components, method, subject_idx):
    return os.path.join(results_root, timestamp, config_tag(n_components, method),
                        f"subject_{subject_idx:04d}")


def adj_src_tgt_to_cg(A):
    """[source, target] binary matrix -> gunfolds CG (1-based keys), no self-loops."""
    n = A.shape[0]
    cg = {i + 1: {} for i in range(n)}
    for s in range(n):
        for t in range(n):
            if s != t and A[s, t]:
                cg[s + 1][t + 1] = 1
    return cg


def clean_adj(A):
    """Binarise, zero the diagonal, return int [source, target] adjacency."""
    A = (np.asarray(A) > 0).astype(int)
    np.fill_diagonal(A, 0)
    return A


def build_subject_info(subject_idx, label, method, n_components,
                       comp_indices, comp_names, A_src_tgt):
    """
    Build the single-solution result payload, byte-for-byte compatible with the
    PCMCI/GCM branches of refactored_fmri_experiment_large.run_single_subject so
    refactored_analyze_fmri_experiment.py treats it identically.
    """
    adj = clean_adj(A_src_tgt)
    cg = adj_src_tgt_to_cg(adj)
    feats = dict(n_solutions=1, ess=1.0, cost_spread=0.0, mean_edge_entropy=0.0,
                 penumbra=0.0, stable_core=float((adj > 0).mean()),
                 map_u=1, mean_u=1.0)
    return {
        "subject_id": int(subject_idx),
        "group": int(label),
        "method": method,
        "config_tag": config_tag(n_components, method),
        "n_components": int(n_components),
        "scc_strategy": "none",
        "comp_indices": list(map(int, comp_indices)),
        "comp_names": list(comp_names),
        "domains": [INDEX_TO_DOMAIN.get(int(c), "?") for c in comp_indices],
        "selection_mode": "single",
        "delta_band": None, "max_keep": 1, "temperature": None,
        "map_u_only": False, "gt_density_mode": None, "gt_density": None,
        "solutions": [{
            "solution_idx": 1, "cost": 0.0, "undersampling": None,
            "graph": cg, "adj": adj.tolist(),
        }],
        "posterior": adj.astype(float).tolist(),
        "features": feats,
        "num_solutions": 1,
        "bootstrap_stability": None, "bootstrap_B": 0,
        "g_estimated": cg,
    }


def save_subject_info(info, results_root, timestamp):
    d = subject_dir(results_root, timestamp, info["n_components"],
                    info["method"], info["subject_id"])
    os.makedirs(d, exist_ok=True)
    zkl.save(info, os.path.join(d, "result.zkl"))
    n_edges = int(np.sum(np.array(info["posterior"]) > 0))
    print(f"[{info['config_tag']}] subject {info['subject_id']} "
          f"(group {info['group']}): {n_edges} edges -> {d}", flush=True)
    return d


def load_data(args):
    npz = np.load(args.data_path)
    data = npz["data"]                       # [n_subjects, T, 53]
    labels = get_labels(npz)                 # [n_subjects]
    comp_indices = get_comp_indices(args.n_components)
    comp_names = get_comp_names(comp_indices)
    return data, labels, comp_indices, comp_names


# ---------------------------------------------------------------------------
# FASK  (py-tetrad: jpype + tetrad jar)  -- per-subject, pure Python
# ---------------------------------------------------------------------------

import re

_FASK_NODE_RE = re.compile(r"Graph Nodes:\s*\n([^\n]+)")
_FASK_EDGE_RE = re.compile(r"(\w+)\s*-->\s*(\w+)")


def _start_jvm(tetrad_jar, pytetrad_path):
    """Start the JVM once with the tetrad jar; add py-tetrad to sys.path."""
    pytetrad_path = os.path.expanduser(pytetrad_path)
    if pytetrad_path and pytetrad_path not in sys.path:
        sys.path.insert(0, pytetrad_path)
    import jpype
    if not jpype.isJVMStarted():
        jar = os.path.expanduser(tetrad_jar)
        if not os.path.isfile(jar):
            raise FileNotFoundError(
                f"tetrad jar not found at '{jar}'. Pass --tetrad_jar or set "
                f"TETRAD_JAR (legacy default: resources/tetrad-current.jar).")
        jpype.startJVM(classpath=[jar])
    # py-tetrad exposes either `tools.TetradSearch` (legacy layout) or
    # `pytetrad.tools.TetradSearch` (pip layout) -- try both.
    try:
        import tools.TetradSearch as TetradSearch
    except ImportError:
        from pytetrad.tools import TetradSearch  # type: ignore
    return TetradSearch


def _fask_string_to_adj(graph_string, comp_names):
    """
    Parse tetrad's text graph into a [source, target] adjacency over comp_names.
    A line "A --> B" means a directed edge A -> B, so adj[idx(A), idx(B)] = 1.
    (Only directed '-->' edges are taken, matching the legacy FASK pipeline.)
    """
    idx = {name: i for i, name in enumerate(comp_names)}
    n = len(comp_names)
    A = np.zeros((n, n), dtype=int)
    for src, dst in _FASK_EDGE_RE.findall(graph_string):
        if src in idx and dst in idx:
            A[idx[src], idx[dst]] = 1
    return A


def run_fask_subject(ts_2d, comp_names, TetradSearch, alpha=0.05,
                     left_right_rule=1):
    """
    Run FASK on one subject's [T, N] series and return a [source, target] adj.
    Mirrors legacy slurm_FASK_time_undersampling_data.py: SEM-BIC score +
    Fisher-Z test, run_fask(alpha, left_right_rule).
    """
    import pandas as pd
    df = pd.DataFrame(np.asarray(ts_2d, dtype=float), columns=list(comp_names))
    search = TetradSearch.TetradSearch(df)
    search.set_verbose(False)
    search.use_sem_bic()
    search.use_fisher_z(alpha=alpha)
    search.run_fask(alpha=alpha, left_right_rule=left_right_rule)
    return _fask_string_to_adj(str(search.get_string()), comp_names)


def stage_fask(args):
    data, labels, comp_indices, comp_names = load_data(args)
    n_subj = data.shape[0]
    TetradSearch = _start_jvm(args.tetrad_jar, args.pytetrad_path)

    subjects = (range(n_subj) if args.all_subjects
                else [args.subject_idx])
    for s in subjects:
        if s < 0 or s >= n_subj:
            raise ValueError(f"subject_idx {s} out of range [0, {n_subj - 1}]")
        ts_2d = data[s][:, comp_indices]                     # [T, N]
        assert ts_2d.shape[1] == len(comp_indices), \
            f"axis swap: {ts_2d.shape}, expected (T, {len(comp_indices)})"
        try:
            A = run_fask_subject(ts_2d, comp_names, TetradSearch,
                                 alpha=args.fask_alpha,
                                 left_right_rule=args.fask_left_right_rule)
        except Exception as e:
            print(f"  WARN FASK subject {s}: {e}; saving empty graph.", flush=True)
            A = np.zeros((len(comp_indices),) * 2, dtype=int)
        info = build_subject_info(s, labels[s], "FASK", args.n_components,
                                  comp_indices, comp_names, A)
        save_subject_info(info, args.results_root, args.timestamp)


# ---------------------------------------------------------------------------
# MVGC / MVAR  (MATLAB bridge)
# ---------------------------------------------------------------------------

def stage_export_matlab(args):
    """Write <workdir>/N<n>/input.mat with all subjects' data for MATLAB."""
    from scipy.io import savemat
    data, labels, comp_indices, comp_names = load_data(args)
    sub = data[:, :, comp_indices].astype(float)             # [nsubj, T, N]
    ndir = os.path.join(args.workdir, f"N{args.n_components}")
    os.makedirs(ndir, exist_ok=True)
    savemat(os.path.join(ndir, "input.mat"), {
        "data": sub,                                         # [nsubj, T, N]
        "labels": np.asarray(labels).astype(int).reshape(-1),
        "comp_indices": np.asarray(comp_indices).astype(int).reshape(-1),
        "comp_names": np.array(comp_names, dtype=object),
        "n_components": int(args.n_components),
    })
    # Sidecar so collect can run without re-reading the npz if desired.
    with open(os.path.join(ndir, "meta.json"), "w") as fh:
        json.dump({"n_subjects": int(sub.shape[0]), "T": int(sub.shape[1]),
                   "n_components": int(args.n_components),
                   "comp_indices": list(map(int, comp_indices)),
                   "comp_names": list(comp_names),
                   "labels": list(map(int, np.asarray(labels).reshape(-1)))},
                  fh, indent=2)
    print(f"[{args.method} export] wrote {os.path.join(ndir, 'input.mat')} "
          f"({sub.shape[0]} subjects, T={sub.shape[1]}, N={sub.shape[2]})\n"
          f"  next: run MATLAB  baselines_{args.method.lower()}"
          f"('{args.workdir}', {args.n_components})", flush=True)


def stage_collect_matlab(args):
    """
    Read <workdir>/N<n>/<METHOD>/sig_<s>.mat (each holding A_src_tgt, a
    [source, target] binary matrix) and write per-subject result.zkl.
    """
    from scipy.io import loadmat
    data, labels, comp_indices, comp_names = load_data(args)
    n_subj = data.shape[0]
    sigdir = os.path.join(args.workdir, f"N{args.n_components}", args.method)
    if not os.path.isdir(sigdir):
        raise FileNotFoundError(
            f"MATLAB output dir not found: {sigdir}. Run the MATLAB step "
            f"(baselines_{args.method.lower()}) before --stage collect.")
    n = len(comp_indices)
    missing = 0
    for s in range(n_subj):
        f = os.path.join(sigdir, f"sig_{s:04d}.mat")
        if os.path.isfile(f):
            m = loadmat(f)
            key = "A_src_tgt" if "A_src_tgt" in m else _first_matrix_key(m)
            A = clean_adj(np.array(m[key]))
            if A.shape != (n, n):
                raise ValueError(f"{f}: A_src_tgt shape {A.shape} != ({n},{n})")
        else:
            missing += 1
            A = np.zeros((n, n), dtype=int)
        info = build_subject_info(s, labels[s], args.method, args.n_components,
                                  comp_indices, comp_names, A)
        save_subject_info(info, args.results_root, args.timestamp)
    if missing:
        print(f"  NOTE: {missing}/{n_subj} subjects had no MATLAB sig file "
              f"(saved empty graphs).", flush=True)


def _first_matrix_key(m):
    for k, v in m.items():
        if not k.startswith("__") and hasattr(v, "shape") and v.ndim == 2:
            return k
    raise KeyError("no 2-D matrix found in MATLAB .mat output")


# ---------------------------------------------------------------------------
# GIMME  (R `gimme` package, pooled group-level run)
# ---------------------------------------------------------------------------

def _gimme_io_dirs(args):
    base = os.path.join(args.workdir, f"N{args.n_components}")
    return os.path.join(base, "gimme_in"), os.path.join(base, "gimme_out")


def stage_export_gimme(args):
    """
    Write one CSV per subject (rows = time, cols = V1..VN) into gimme_in/ plus a
    mapping file (filename -> subject_idx, group).  gimme() is then run ONCE
    over that folder (pooled), estimating a shared group structure.
    """
    import pandas as pd
    data, labels, comp_indices, comp_names = load_data(args)
    sub = data[:, :, comp_indices].astype(float)             # [nsubj, T, N]
    in_dir, _ = _gimme_io_dirs(args)
    os.makedirs(in_dir, exist_ok=True)
    cols = [f"V{k + 1}" for k in range(len(comp_indices))]    # gimme-safe names
    mapping = {}
    for s in range(sub.shape[0]):
        fn = f"sub_{s:04d}.csv"
        pd.DataFrame(sub[s], columns=cols).to_csv(
            os.path.join(in_dir, fn), index=False)
        mapping[fn] = {"subject_idx": int(s), "group": int(labels[s])}
    meta = {"mapping": mapping, "var_cols": cols,
            "comp_indices": list(map(int, comp_indices)),
            "comp_names": list(comp_names),
            "n_components": int(args.n_components)}
    with open(os.path.join(os.path.dirname(in_dir), "gimme_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[GIMME export] wrote {sub.shape[0]} CSVs to {in_dir}\n"
          f"  next: Rscript baselines_run_gimme.R '{in_dir}' "
          f"'{_gimme_io_dirs(args)[1]}'", flush=True)


def _var_idx(name, n):
    """'V7' / 'V7lag' -> 0-based index 6; returns None if unparseable."""
    base = name[:-3] if name.endswith("lag") else name
    m = re.fullmatch(r"V(\d+)", base)
    if not m:
        return None
    k = int(m.group(1)) - 1
    return k if 0 <= k < n else None


def stage_collect_gimme(args):
    """
    Parse gimme's indivPathEstimates.csv (long format: file, lhs, op, rhs, ...)
    into a per-subject [source, target] adjacency and write result.zkl.

    A row "lhs ~ rhs" is a regression outcome ~ predictor, i.e. a directed path
    predictor -> outcome.  Both contemporaneous (rhs='Vk') and lagged
    (rhs='Vklag') predictors map to the same directed edge src -> tgt (matching
    the legacy GIMME reader, which OR-ed the beta and phi matrices).  Self
    paths (the AR term Vk lag -> Vk) fall on the diagonal and are dropped.
    """
    import pandas as pd
    in_dir, out_dir = _gimme_io_dirs(args)
    meta_path = os.path.join(os.path.dirname(in_dir), "gimme_meta.json")
    with open(meta_path) as fh:
        meta = json.load(fh)
    mapping = meta["mapping"]
    comp_indices = meta["comp_indices"]
    comp_names = meta["comp_names"]
    n = len(comp_indices)

    est_csv = os.path.join(out_dir, "indivPathEstimates.csv")
    if not os.path.isfile(est_csv):
        raise FileNotFoundError(
            f"gimme output not found: {est_csv}. Run "
            f"`Rscript baselines_run_gimme.R '{in_dir}' '{out_dir}'` first.")
    df = pd.read_csv(est_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    file_col = _pick(df.columns, ["file", "id", "subject"])
    lhs_col = _pick(df.columns, ["lhs"])
    rhs_col = _pick(df.columns, ["rhs"])
    op_col = _pick(df.columns, ["op"], required=False)

    # Group rows by subject file stem.
    df["_stem"] = df[file_col].astype(str).apply(
        lambda x: os.path.splitext(os.path.basename(x))[0])

    for fn, info_map in mapping.items():
        stem = os.path.splitext(fn)[0]
        rows = df[df["_stem"] == stem]
        if op_col is not None:
            rows = rows[rows[op_col].astype(str).str.strip() == "~"]
        A = np.zeros((n, n), dtype=int)
        for _, r in rows.iterrows():
            tgt = _var_idx(str(r[lhs_col]).strip(), n)        # outcome
            src = _var_idx(str(r[rhs_col]).strip(), n)        # predictor
            if src is None or tgt is None or src == tgt:
                continue
            A[src, tgt] = 1                                    # predictor -> outcome
        info = build_subject_info(info_map["subject_idx"], info_map["group"],
                                  "GIMME", args.n_components, comp_indices,
                                  comp_names, A)
        save_subject_info(info, args.results_root, args.timestamp)


def _pick(cols, candidates, required=True):
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"expected one of {candidates} in gimme columns {list(cols)}")
    return None


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def parse_arguments():
    p = argparse.ArgumentParser(description="GIMME/MVAR/MVGC/FASK baselines on "
                                "FBIRN, saved in the refactored result.zkl format.")
    p.add_argument("--method", required=True, choices=METHODS)
    p.add_argument("--stage", default=None,
                   choices=["run", "export", "collect"],
                   help="FASK uses 'run'; MVGC/MVAR/GIMME use 'export' then "
                        "'collect' (with the MATLAB/R step in between).")
    p.add_argument("--n_components", type=int, default=10,
                   choices=[10, 13, 14, 15, 20, 53])
    p.add_argument("--subject_idx", type=int, default=0,
                   help="FASK single-subject mode (SLURM array task).")
    p.add_argument("--all_subjects", action="store_true",
                   help="FASK: run every subject in one process (one JVM).")
    p.add_argument("--timestamp", default=None)
    p.add_argument("--results_root", default="fbirn_results_refactored",
                   help="Write under <results_root>/<timestamp>/<config_tag>/ "
                        "(default matches RASL/PCMCI so analysis groups them).")
    p.add_argument("--data_path", default="../fbirn/fbirn_sz_data.npz")
    p.add_argument("--workdir", default="baselines_work",
                   help="Scratch dir for MATLAB .mat / GIMME CSV I/O.")

    # FASK / py-tetrad
    p.add_argument("--tetrad_jar", default=os.environ.get(
        "TETRAD_JAR", "resources/tetrad-current.jar"))
    p.add_argument("--pytetrad_path", default=os.environ.get(
        "PYTETRAD_PATH", "~/tread/py-tetrad"))
    p.add_argument("--fask_alpha", type=float, default=0.05)
    p.add_argument("--fask_left_right_rule", type=int, default=1)
    return p.parse_args()


def main():
    args = parse_arguments()
    args.timestamp = args.timestamp or datetime.now().strftime("%m%d%Y%H%M%S")

    # Default stage per method.
    if args.stage is None:
        args.stage = "run" if args.method == "FASK" else "export"

    print("=" * 78)
    print(f"BASELINE EXPERIMENT  method={args.method}  stage={args.stage}  "
          f"N={args.n_components}")
    print(f"  results -> {args.results_root}/{args.timestamp}/"
          f"{config_tag(args.n_components, args.method)}/")
    print("=" * 78, flush=True)

    if args.method == "FASK":
        if args.stage != "run":
            raise SystemExit("FASK only supports --stage run.")
        stage_fask(args)
    elif args.method in ("MVGC", "MVAR"):
        if args.stage == "export":
            stage_export_matlab(args)
        elif args.stage == "collect":
            stage_collect_matlab(args)
        else:
            raise SystemExit(f"{args.method} supports --stage export|collect "
                             f"(MATLAB runs in between).")
    elif args.method == "GIMME":
        if args.stage == "export":
            stage_export_gimme(args)
        elif args.stage == "collect":
            stage_collect_gimme(args)
        else:
            raise SystemExit("GIMME supports --stage export|collect "
                             "(Rscript baselines_run_gimme.R runs in between).")


if __name__ == "__main__":
    main()
