# run_all_subjects_roebroeck.py
import os, numpy as np, pandas as pd, argparse, glob
import networkx as nx
import matplotlib.pyplot as plt
from roebroeck_gcm import run_roebroeck_gcm

# ---------- config ----------
npz_path   = "./fbirn/fbirn_sz_data.npz"
comp_idx   = [25, 29, 35, 44, 45, 46]       # ICA indices
label_map  = {25:"rPPC", 29:"rFIC", 35:"rDLPFC", 44:"ACC", 45:"PCC", 46:"VMPFC"}
names      = [label_map[k] for k in comp_idx]

tr_sec     = 2.0
n_cycles   = 0
remove_linear = False
alpha      = 0.01
pmax       = 8
n_boot     = 200
surr_mode  = "block"
block_len  = 16
seed       = 0
out_base   = "./gcm_roebroeck"              # figures -> ./gcm_roebroeck/figures ; CSVs -> ./gcm_roebroeck/csv

# ---------- args ----------
parser = argparse.ArgumentParser(description="Run Roebroeck GCM per-subject and/or aggregate saved CSVs.")
parser.add_argument('-S', '--subject', type=int, default=1, help='Subject index to run only this subject')
parser.add_argument('-A', '--alpha', type=int, default=50, help='Alpha x1000 (e.g., 50 -> 0.05)')
parser.add_argument('--aggregate-csvs', type=str, default=None, help='Directory containing per-subject *Adj_labeled.csv to aggregate')
args = parser.parse_args()

if args.alpha is not None:
    alpha = args.alpha / 1000.0

# ---------- io helpers ----------


def fixed_circle_positions(names):
    # names order defines placement. First node at top, then clockwise.
    N = len(names)
    angles = np.linspace(-np.pi/2, 3*np.pi/2, N, endpoint=False)
    return {i: (float(np.cos(a)), float(np.sin(a))) for i, a in enumerate(angles)}
def ensure_dirs(base):
    fig_dir = os.path.join(base, "figures")
    csv_dir = os.path.join(base, "csv")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return fig_dir, csv_dir


# ---------- aggregation helper ----------
def aggregate_from_csv_dir(csv_dir: str, out_base_dir: str):
    # Find all subject adjacency CSVs
    files = sorted(glob.glob(os.path.join(csv_dir, '*Adj_labeled.csv')))
    if not files:
        print(f"No *Adj_labeled.csv files found in {csv_dir}")
        return
    # Initialize with the first
    first = pd.read_csv(files[0], index_col=0)
    names = list(first.columns)
    N = len(names)
    hits = np.zeros((N, N), dtype=int)
    # Sum across subjects
    for f in files:
        df = pd.read_csv(f, index_col=0)
        # Reorder if needed
        df = df.loc[names, names]
        hits += df.to_numpy().astype(int)
    S_total = len(files)
    rate = hits / float(S_total)

    fig_dir, _ = ensure_dirs(out_base_dir)
    grp_prefix = os.path.join(csv_dir, 'group')
    pd.DataFrame(hits, index=names, columns=names).to_csv(grp_prefix + '_edge_hits.csv')
    pd.DataFrame(rate, index=names, columns=names).to_csv(grp_prefix + '_edge_rate.csv')

    # Plot
    out_png = os.path.join(fig_dir, 'group_edge_frequency.png')
    save_group_plot(rate, names, out_png)
    print(f"Aggregated {S_total} subjects. Group plot -> {out_png}")
    print(f"Group CSVs -> {grp_prefix}_edge_{{hits,rate}}.csv")

def save_group_plot(edge_weight, node_names, out_png):
    N = edge_weight.shape[0]
    G = nx.DiGraph()
    for i in range(N):
        G.add_node(i, label=node_names[i])
    W = edge_weight / (edge_weight.max() + 1e-12)
    for i in range(N):
        for j in range(N):
            if i != j and edge_weight[i, j] > 0:
                G.add_edge(i, j, weight=float(W[i, j]), raw=float(edge_weight[i, j]))
    pos = fixed_circle_positions(node_names)  # names = ["rPPC","rFIC","rDLPFC","ACC","PCC","VMPFC"]
    plt.figure(figsize=(6, 5))

    NODE_SIZE = 700
    nx.draw_networkx_nodes(G, pos, node_size=NODE_SIZE)
    nx.draw_networkx_labels(G, pos, labels={k: node_names[k] for k in range(N)}, font_size=9)

    # Map normalized edge weights (0..1) to visually separated widths
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    if weights:
        w_min, w_max = min(weights), max(weights)
        if w_min == w_max:
            widths = [10.0] * len(weights)
        else:
            min_thick, max_thick = 1, 10
            k = 5  # larger -> more separation of larger weights
            denom = np.exp(k) - 1.0
            widths = []
            for w in weights:
                t = (w - w_min) / (w_max - w_min)
                s = (np.exp(k * t) - 1.0) / denom
                widths.append(min_thick + (max_thick - min_thick) * s)
    else:
        widths = []

    # Ensure arrowheads are sharp and reach node borders
    max_w = max(widths) if widths else 1.0
    arrowsize = int(max(18, np.ceil(2.5 * max_w)))
    margin = 2.0  # points

    # Draw edges: single-headed; if both directions exist, curve both with same rad
    widths_map = {edge: w for edge, w in zip(G.edges(), widths)}
    drawn = set()
    for (u, v) in G.edges():
        if (u, v) in drawn:
            continue
        has_back = G.has_edge(v, u)
        if has_back:
            w_uv = widths_map.get((u, v), 1.0)
            w_vu = widths_map.get((v, u), 1.0)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=w_uv,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=arrowsize,
                min_source_margin=margin,
                min_target_margin=margin,
                connectionstyle='arc3,rad=0.25',
            )
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(v, u)],
                width=w_vu,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=arrowsize,
                min_source_margin=margin,
                min_target_margin=margin,
                connectionstyle='arc3,rad=0.25',
            )
            drawn.add((u, v)); drawn.add((v, u))
        else:
            w_uv = widths_map.get((u, v), 1.0)
            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=w_uv,
                arrows=True,
                arrowstyle='-|>',
                arrowsize=arrowsize,
                min_source_margin=margin,
                min_target_margin=margin,
                connectionstyle='arc3,rad=0.0',
            )

    plt.title("Group GCM: edge frequency")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------- load data ----------
# If only aggregating from existing CSVs, do that and exit
if args.aggregate_csvs:
    aggregate_from_csv_dir(args.aggregate_csvs, out_base)
    import sys; sys.exit(0)

npz = np.load(npz_path)
data = npz["data"]            # shape: (subjects, time, components)
S, T, C = data.shape
print(f"Subjects={S}  T={T}  Components={C}")

# ---------- per-subject runs ----------
fig_dir, csv_dir = ensure_dirs(out_base)
N = len(comp_idx)
edge_hits = np.zeros((N, N), dtype=int)
Fdiff_sum = np.zeros((N, N), dtype=float)

# Choose subjects to run
if args.subject is not None:
    if args.subject < 0 or args.subject >= S:
        raise ValueError(f"Subject index out of range: {args.subject} (0..{S-1})")
    subject_iter = [args.subject]
else:
    subject_iter = range(S)

for s in subject_iter:
    Xs = data[s, :, comp_idx].T                 # (T, 6)
    tag = f"subj{s}_IC{','.join(map(str, comp_idx))}"
    res = run_roebroeck_gcm(
        Xs, tr_sec=tr_sec,
        n_cycles=n_cycles, remove_linear=remove_linear,
        alpha=alpha, pmax=pmax, n_boot=n_boot,
        surr_mode=surr_mode, block_len=block_len,
        seed=seed, fdr=False,
        names=names, make_plot=True,
        out_base=out_base, run_tag=tag
    )
    # accumulate
    edge_hits += res["Adj"].astype(int)
    Fdiff_sum += res["Fdiff"]

    # also save subject-level matrices with readable row/col labels
    prefix = res["csv_prefix"]  # already unique
    pd.DataFrame(res["Fdiff"], index=names, columns=names).to_csv(prefix + "Fdiff_labeled.csv")
    pd.DataFrame(res["Pdiff"], index=names, columns=names).to_csv(prefix + "Pdiff_labeled.csv")
    pd.DataFrame(res["Adj"].astype(int), index=names, columns=names).to_csv(prefix + "Adj_labeled.csv")
    print(f"Subject {s}: fig -> {res['fig_path']}")

# ---------- group summaries ----------
if args.subject is None:
    edge_rate = edge_hits / S
    Fdiff_mean = Fdiff_sum / S

    grp_prefix = os.path.join(csv_dir, "group")
    pd.DataFrame(edge_hits, index=names, columns=names).to_csv(grp_prefix + "_edge_hits.csv")
    pd.DataFrame(edge_rate, index=names, columns=names).to_csv(grp_prefix + "_edge_rate.csv")
    pd.DataFrame(Fdiff_mean, index=names, columns=names).to_csv(grp_prefix + "_Fdiff_mean.csv")

    # group-level plot: edge width ~ frequency
    grp_png = os.path.join(fig_dir, "group_edge_frequency.png")
    save_group_plot(edge_rate, names, grp_png)
    print('Group plot ->', grp_png)
    print('Group CSVs ->', grp_prefix + '_edge_{hits,rate}.csv, _Fdiff_mean.csv')
else:
    print(f"Completed subject {subject_iter[0]} only. CSVs and figure saved above.")