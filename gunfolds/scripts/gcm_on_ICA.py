# run_all_subjects_roebroeck.py
import os, numpy as np, pandas as pd
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
alpha      = 0.05
pmax       = 8
n_boot     = 200
surr_mode  = "block"
block_len  = 16
seed       = 0
out_base   = "./gcm_roebroeck"              # figures -> ./gcm_roebroeck/figures ; CSVs -> ./gcm_roebroeck/csv

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
    pos = fixed_circle_positions(names)  # names = ["rPPC","rFIC","rDLPFC","ACC","PCC","VMPFC"]
    plt.figure(figsize=(6, 5))
    nx.draw_networkx_nodes(G, pos, node_size=700)
    nx.draw_networkx_labels(G, pos, labels={k: node_names[k] for k in range(N)}, font_size=9)
    widths = [1 + 6 * G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, arrows=True, arrowstyle='-|>', arrowsize=12)
    plt.title("Group GCM: edge frequency")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

# ---------- load data ----------
npz = np.load(npz_path)
data = npz["data"]            # shape: (subjects, time, components)
S, T, C = data.shape
print(f"Subjects={S}  T={T}  Components={C}")

# ---------- per-subject runs ----------
fig_dir, csv_dir = ensure_dirs(out_base)
N = len(comp_idx)
edge_hits = np.zeros((N, N), dtype=int)
Fdiff_sum = np.zeros((N, N), dtype=float)

for s in range(S):
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
edge_rate = edge_hits / S
Fdiff_mean = Fdiff_sum / S

grp_prefix = os.path.join(csv_dir, "group")
pd.DataFrame(edge_hits, index=names, columns=names).to_csv(grp_prefix + "_edge_hits.csv")
pd.DataFrame(edge_rate, index=names, columns=names).to_csv(grp_prefix + "_edge_rate.csv")
pd.DataFrame(Fdiff_mean, index=names, columns=names).to_csv(grp_prefix + "_Fdiff_mean.csv")

# group-level plot: edge width ~ frequency
grp_png = os.path.join(fig_dir, "group_edge_frequency.png")
save_group_plot(edge_rate, names, grp_png)
print("Group plot ->", grp_png)
print("Group CSVs ->", grp_prefix + "_edge_{hits,rate}.csv, _Fdiff_mean.csv")