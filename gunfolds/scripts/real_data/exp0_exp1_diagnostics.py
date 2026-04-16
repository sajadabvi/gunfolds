"""
Experiment 0: COMPLETED -- pp.DataFrame(ts_2d) is correct, .T is wrong.
Experiment 1: Verify Glag2CG transpose correctness with known synthetic ground truth.

Run from: gunfolds/scripts/real_data/
    conda activate gunfolds
    python exp0_exp1_diagnostics.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))

import numpy as np
import tigramite.data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

from gunfolds import conversions as cv
from gunfolds.conversions import adjs2graph


# ========================================================================
# EXPERIMENT 0 RECAP (already confirmed)
# ========================================================================
print("=" * 70)
print("EXPERIMENT 0 RECAP")
print("=" * 70)
print("CONFIRMED: pp.DataFrame expects (T, N).")
print("  pp.DataFrame(ts_2d)   where ts_2d is [T,N] => CORRECT")
print("  pp.DataFrame(ts_2d.T) where ts_2d is [T,N] => WRONG (swaps T and N)")
print("The current codebase uses .T everywhere -- this is a bug.")


# ========================================================================
# EXPERIMENT 1: Glag2CG Transpose Verification
# ========================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: Glag2CG Transpose Verification")
print("=" * 70)

# 3-node chain: var0(t-1) --> var1(t), var1(t-1) --> var2(t)
np.random.seed(42)
T = 5000
x1 = np.random.randn(T)
x2 = np.zeros(T)
x3 = np.zeros(T)
for t in range(1, T):
    x2[t] = 0.7 * x1[t - 1] + 0.1 * np.random.randn()
    x3[t] = 0.7 * x2[t - 1] + 0.1 * np.random.randn()

data = np.column_stack([x1, x2, x3])  # [T, 3]
print(f"Data shape: {data.shape}")
print("Ground truth: var0-->var1, var1-->var2")
print("Gunfolds (1-indexed): {1: {2: 1}, 2: {3: 1}, 3: {}}")

# Use the CORRECT convention: pp.DataFrame(data) where data is [T, N]
df = pp.DataFrame(data)
print(f"\nDataFrame T={df.T}, N={df.N}")
pcmci_obj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
res = pcmci_obj.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)

print(f"\nRaw graph[:,:,1] (lag-1):")
print(f"  {res['graph'][:, :, 1]}")
print(f"Raw graph[:,:,0] (contemp):")
print(f"  {res['graph'][:, :, 0]}")
print(f"val_matrix[:,:,1]:")
print(f"  {np.round(res['val_matrix'][:, :, 1], 4)}")
print(f"p_matrix[:,:,1]:")
print(f"  {np.round(res['p_matrix'][:, :, 1], 6)}")

print(f"\nInterpreting graph[i,j,tau]: link from var_j(t-tau) to var_i(t)")
for i in range(3):
    for j in range(3):
        if res['graph'][i, j, 1] == '-->':
            print(f"  graph[{i},{j},1] = '-->' => var{j}(t-1) --> var{i}(t)")

# Canonical Glag2CG (with np.transpose)
g_canon, A_c, B_c = cv.Glag2CG(res)
print(f"\nCanonical Glag2CG (with transpose):  {g_canon}")

# No-transpose version (as in local copies)
graph_array = res['graph']
de = np.where(graph_array == '-->', 1, 0).astype(int)
be = np.where(graph_array == 'o-o', 1, 0).astype(int)
g_no_t = adjs2graph(de[:, :, 1], be[:, :, 0])
print(f"No-transpose Glag2CG:                 {g_no_t}")
print(f"Expected:                              {{1: {{2: 1}}, 2: {{3: 1}}, 3: {{}}}}")

match_canon = (g_canon == {1: {2: 1}, 2: {3: 1}, 3: {}})
match_no_t = (g_no_t == {1: {2: 1}, 2: {3: 1}, 3: {}})
print(f"\nCanonical matches expected: {match_canon}")
print(f"No-transpose matches expected: {match_no_t}")


# ========================================================================
# EXPERIMENT 1b: Asymmetric 2-node case
# ========================================================================
print("\n--- Experiment 1b: Asymmetric 2-node case ---")
np.random.seed(123)
T = 5000
a = np.random.randn(T)
b = np.zeros(T)
for t in range(1, T):
    b[t] = 0.8 * a[t - 1] + 0.05 * np.random.randn()

data2 = np.column_stack([a, b])  # [T, 2]
print(f"Data shape: {data2.shape}")
print("Ground truth: var0-->var1 ONLY")
print("Gunfolds (1-indexed): {1: {2: 1}, 2: {}}")

df = pp.DataFrame(data2)
print(f"DataFrame T={df.T}, N={df.N}")
pcmci_obj = PCMCI(dataframe=df, cond_ind_test=ParCorr(), verbosity=0)
res = pcmci_obj.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)

print(f"\nRaw graph[:,:,1]: {res['graph'][:, :, 1]}")
for i in range(2):
    for j in range(2):
        if res['graph'][i, j, 1] == '-->':
            print(f"  graph[{i},{j},1] = '-->' => var{j}(t-1) --> var{i}(t)")

g_canon, _, _ = cv.Glag2CG(res)
print(f"\nCanonical (transpose):  {g_canon}")

de = np.where(res['graph'] == '-->', 1, 0).astype(int)
be = np.where(res['graph'] == 'o-o', 1, 0).astype(int)
g_no_t = adjs2graph(de[:, :, 1], be[:, :, 0])
print(f"No-transpose:           {g_no_t}")
print(f"Expected:               {{1: {{2: 1}}, 2: {{}}}}")

match_canon = (g_canon == {1: {2: 1}, 2: {}})
match_no_t = (g_no_t == {1: {2: 1}, 2: {}})
print(f"\nCanonical matches expected: {match_canon}")
print(f"No-transpose matches expected: {match_no_t}")


# ========================================================================
# Summary
# ========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("1. pp.DataFrame convention: pp.DataFrame(ts_2d) is CORRECT (not .T)")
print("2. Glag2CG transpose: check results above for which version matches")
print("=" * 70)
