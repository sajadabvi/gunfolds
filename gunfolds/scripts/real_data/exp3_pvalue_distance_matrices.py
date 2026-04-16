"""
Experiment 3: Compare binary vs p-value-weighted distance matrices for RASL.

Runs on a few subjects and prints the distance matrices side by side.

Run from: gunfolds/scripts/real_data/
    conda activate gunfolds
    python exp3_pvalue_distance_matrices.py

NOTE: Adjust the pp.DataFrame() call based on Experiment 0 results.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr
from gunfolds import conversions as cv
from component_config import get_comp_indices

N_SUBJECTS_TO_TEST = 3


def run_pcmci_full(ts_2d, tau_max=1, alpha_level=0.1):
    """Return g_estimated, A, B, p_matrix, val_matrix."""
    # Exp 0 confirmed: pp.DataFrame expects (T, N), ts_2d is already [T, N]
    df = pp.DataFrame(ts_2d)
    pcmci = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
    res = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=None, alpha_level=alpha_level)
    g_est, A, B = cv.Glag2CG(res)
    return g_est, A, B, res['p_matrix'], res['val_matrix']


def main():
    comp_indices = get_comp_indices(10)
    npzfile = np.load("../fbirn/fbirn_sz_data.npz")
    data = npzfile["data"]
    print(f"Data shape: {data.shape}")
    print(f"Components: {comp_indices}")

    for s in range(N_SUBJECTS_TO_TEST):
        print(f"\n{'='*70}")
        print(f"Subject {s}")
        print(f"{'='*70}")

        ts_2d = data[s][:, comp_indices]  # [T, N]
        g_est, A, B, p_matrix, val_matrix = run_pcmci_full(ts_2d)

        print(f"\np_matrix shape: {p_matrix.shape}")
        print(f"val_matrix shape: {val_matrix.shape}")
        print(f"g_estimated: {g_est}")

        N = len(g_est)
        MAXCOST = 50

        # --- Current DD: val_matrix-based ---
        a_max = np.abs(A).max()
        b_max = np.abs(B).max()
        adj_dir = cv.graph2adj(g_est)
        adj_bid = cv.graph2badj(g_est)

        if a_max > 0:
            DD_current = (np.abs((np.abs(A / a_max) + (adj_dir - 1)) * MAXCOST)).astype(int)
        else:
            DD_current = (np.abs((adj_dir - 1) * MAXCOST)).astype(int)

        if b_max > 0:
            BD_current = (np.abs((np.abs(B / b_max) + (adj_bid - 1)) * MAXCOST)).astype(int)
        else:
            BD_current = (np.abs((adj_bid - 1) * MAXCOST)).astype(int)

        # --- Alternative DD: p-value weighted ---
        # For edges in g_est: use -log10(p) as weight (higher = more confident)
        # For non-edges: keep MAXCOST penalty
        p_lag1 = p_matrix[:, :, 1]
        p_contemp = p_matrix[:, :, 0]

        # Directed distance: use p-values at lag 1
        p_dir_weight = np.where(p_lag1 < 0.1, -np.log10(np.clip(p_lag1, 1e-15, 1.0)), 0)
        p_dir_max = p_dir_weight.max() if p_dir_weight.max() > 0 else 1
        DD_pval = np.where(
            adj_dir == 1,
            (MAXCOST * (1.0 - p_dir_weight / p_dir_max)).astype(int),
            MAXCOST
        )

        # Bidirected distance: use p-values at lag 0
        p_bid_weight = np.where(p_contemp < 0.1, -np.log10(np.clip(p_contemp, 1e-15, 1.0)), 0)
        p_bid_max = p_bid_weight.max() if p_bid_weight.max() > 0 else 1
        BD_pval = np.where(
            adj_bid == 1,
            (MAXCOST * (1.0 - p_bid_weight / p_bid_max)).astype(int),
            MAXCOST
        )

        print(f"\n--- Directed Distance Matrix (DD) ---")
        print(f"DD_current (val_matrix based):\n{DD_current}")
        print(f"\nDD_pval (p-value based):\n{DD_pval}")
        print(f"\nDD difference (pval - current):\n{DD_pval - DD_current}")

        print(f"\n--- Bidirected Distance Matrix (BD) ---")
        print(f"BD_current (val_matrix based):\n{BD_current}")
        print(f"\nBD_pval (p-value based):\n{BD_pval}")
        print(f"\nBD difference (pval - current):\n{BD_pval - BD_current}")

        # Summary stats
        dir_edges = np.sum(adj_dir == 1)
        bid_edges = np.sum(adj_bid == 1)
        print(f"\nSummary:")
        print(f"  Directed edges: {dir_edges}, Bidirected edges: {bid_edges}")
        print(f"  DD_current range on edges: "
              f"[{DD_current[adj_dir==1].min() if dir_edges else 'N/A'}, "
              f"{DD_current[adj_dir==1].max() if dir_edges else 'N/A'}]")
        print(f"  DD_pval range on edges:    "
              f"[{DD_pval[adj_dir==1].min() if dir_edges else 'N/A'}, "
              f"{DD_pval[adj_dir==1].max() if dir_edges else 'N/A'}]")

        # Show raw p-values for detected edges
        print(f"\n  p-values for detected directed edges:")
        for i in range(N):
            for j in range(N):
                if adj_dir[i, j] == 1:
                    pval = p_lag1[i, j]
                    vval = A[i, j] if a_max > 0 else 0
                    print(f"    edge {i+1}->{j+1}: p={pval:.6f}, val={vval:.4f}, "
                          f"DD_cur={DD_current[i,j]}, DD_pval={DD_pval[i,j]}")

    print(f"\n{'='*70}")
    print("DONE -- Please paste the full output back.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
