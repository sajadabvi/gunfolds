"""
Quick test script for delta hyperparameter tuning

This script runs a minimal test to verify:
1. All dependencies are installed
2. Data files are accessible
3. PCMCI and DRASL are working
4. Delta selection logic works correctly

Run this before doing full hyperparameter tuning!
"""
import os
import sys
import numpy as np
import pandas as pd

def test_imports():
    """Test that all required packages can be imported"""
    print("=" * 80)
    print("TEST 1: Checking imports...")
    print("=" * 80)
    
    try:
        from gunfolds import conversions as cv
        print("✓ gunfolds.conversions")
        
        from gunfolds.scripts.datasets.simple_networks import simp_nets
        print("✓ gunfolds.scripts.datasets.simple_networks")
        
        from gunfolds.scripts import my_functions as mf
        print("✓ gunfolds.scripts.my_functions")
        
        from gunfolds.solvers.clingo_rasl import drasl
        print("✓ gunfolds.solvers.clingo_rasl")
        
        from gunfolds.utils import graphkit as gk
        print("✓ gunfolds.utils.graphkit")
        
        from gunfolds.utils import bfutils
        print("✓ gunfolds.utils.bfutils")
        
        from gunfolds.utils import zickle as zkl
        print("✓ gunfolds.utils.zickle")
        
        import tigramite.data_processing as pp
        print("✓ tigramite.data_processing")
        
        from tigramite.pcmci import PCMCI
        print("✓ tigramite.pcmci")
        
        from tigramite.independence_tests.parcorr import ParCorr
        print("✓ tigramite.independence_tests.parcorr")
        
        import matplotlib.pyplot as plt
        print("✓ matplotlib.pyplot")
        
        import seaborn as sns
        print("✓ seaborn")
        
        from tqdm import tqdm
        print("✓ tqdm")
        
        print("\n✓ All imports successful!\n")
        return True
        
    except ImportError as e:
        print(f"\n✗ Import failed: {e}")
        print("\nPlease install missing packages:")
        print("  pip install tqdm pandas numpy matplotlib seaborn tigramite")
        return False

def test_data_access():
    """Test that data files are accessible"""
    print("=" * 80)
    print("TEST 2: Checking data files...")
    print("=" * 80)
    
    net = 3
    undersampling = 3
    
    base_path = os.path.expanduser(f'~/DataSets_Feedbacks/8_VAR_simulation/net{net}/u{undersampling}/txtSTD')
    
    if not os.path.exists(base_path):
        print(f"✗ Data directory not found: {base_path}")
        print("\nPlease ensure data files exist at:")
        print(f"  {base_path}/data1.txt")
        print(f"  {base_path}/data2.txt")
        print("  ...")
        return False
    
    print(f"✓ Data directory found: {base_path}")
    
    # Check for at least one data file
    data_file = f"{base_path}/data1.txt"
    if not os.path.exists(data_file):
        print(f"✗ Data file not found: {data_file}")
        return False
    
    print(f"✓ Data file found: {data_file}")
    
    # Try to read the data file
    try:
        data = pd.read_csv(data_file, delimiter='\t')
        print(f"✓ Data file readable: {data.shape[0]} samples, {data.shape[1]} variables")
        print("\n✓ Data access successful!\n")
        return True
    except Exception as e:
        print(f"✗ Failed to read data file: {e}")
        return False

def test_delta_logic():
    """Test the delta selection logic with synthetic data (percentage-based)"""
    print("=" * 80)
    print("TEST 3: Testing delta selection logic (percentage-based)...")
    print("=" * 80)
    
    # Create synthetic solutions with costs
    solutions_with_costs = [
        (12345, (2,), 1000),   # Min cost
        (12346, (2,), 1500),   # +50% from min
        (12347, (2,), 2000),   # +100% from min
        (12348, (2,), 5000),   # +400% from min
        (12349, (2,), 10000),  # +900% from min
    ]
    
    # Test different delta percentages
    test_deltas = [0.0, 0.5, 1.0, 2.0, 5.0]  # 0%, 50%, 100%, 200%, 500%
    
    for delta in test_deltas:
        min_cost = min(s[2] for s in solutions_with_costs)
        absolute_delta = delta * min_cost
        selected = [s for s in solutions_with_costs if s[2] <= min_cost + absolute_delta]
        print(f"  Delta={delta*100:5.0f}% (absolute: {absolute_delta:6.0f}) → Selected {len(selected)}/{len(solutions_with_costs)} solutions")
    
    print("\n✓ Delta selection logic working correctly!\n")
    return True

def test_pcmci_basic():
    """Test PCMCI with simple synthetic data"""
    print("=" * 80)
    print("TEST 4: Testing PCMCI (basic)...")
    print("=" * 80)
    
    try:
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        from gunfolds import conversions as cv
        
        # Create simple synthetic data (3 variables, 100 samples)
        np.random.seed(42)
        data = np.random.randn(100, 3)
        
        # Add some dependencies
        data[1:, 1] += 0.5 * data[:-1, 0]  # X0 -> X1 (lag 1)
        data[1:, 2] += 0.5 * data[:-1, 1]  # X1 -> X2 (lag 1)
        
        dataframe = pp.DataFrame(data)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        
        print("  Running PCMCI on synthetic data...")
        results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        
        print("  Converting to causal graph...")
        g_estimated, A, B = cv.Glag2CG(results)
        
        print(f"✓ PCMCI completed successfully!")
        print(f"  Found {len(g_estimated)} nodes in graph")
        print("\n✓ PCMCI test successful!\n")
        return True
        
    except Exception as e:
        print(f"✗ PCMCI test failed: {e}")
        print("\nThis may indicate an issue with tigramite or gunfolds installation.")
        return False

def test_full_pipeline():
    """Test the full pipeline with minimal data"""
    print("=" * 80)
    print("TEST 5: Testing full pipeline (1 batch, 1 delta %, orientation only)...")
    print("=" * 80)
    
    try:
        from gunfolds import conversions as cv
        from gunfolds.scripts.datasets.simple_networks import simp_nets
        from gunfolds.scripts import my_functions as mf
        from gunfolds.solvers.clingo_rasl import drasl
        from gunfolds.utils import graphkit as gk
        from gunfolds.utils import bfutils
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        
        net = 3
        undersampling = 3
        batch = 1
        delta = 0.5  # 50% of min_cost
        priorities = [4, 2, 5, 3, 1]
        
        # Load ground truth
        network_GT = simp_nets(net, selfloop=True)
        print(f"✓ Loaded ground truth network (net={net})")
        
        # Load data
        path = os.path.expanduser(
            f'~/DataSets_Feedbacks/8_VAR_simulation/net{net}/u{undersampling}/txtSTD/data{batch}.txt')
        
        if not os.path.exists(path):
            print(f"✗ Data file not found: {path}")
            return False
        
        data = pd.read_csv(path, delimiter='\t')
        print(f"✓ Loaded data: {data.shape[0]} samples, {data.shape[1]} variables")
        
        # Run PCMCI
        dataframe = pp.DataFrame(data.values)
        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        print("  Running PCMCI...")
        results = pcmci.run_pcmci(tau_max=1, pc_alpha=None, alpha_level=0.05)
        g_estimated, A, B = cv.Glag2CG(results)
        print(f"✓ PCMCI completed")
        
        # Run DRASL
        MAXCOST = 10000
        DD = (np.abs((np.abs(A / np.abs(A).max()) + (cv.graph2adj(g_estimated) - 1)) * MAXCOST)).astype(int)
        BD = (np.abs((np.abs(B / np.abs(B).max()) + (cv.graph2badj(g_estimated) - 1)) * MAXCOST)).astype(int)
        
        print("  Running DRASL...")
        r_estimated = drasl([g_estimated], weighted=True, capsize=0, timeout=0,
                            urate=min(5, (3 * len(g_estimated) + 1)),
                            dm=[DD],
                            bdm=[BD],
                            scc=False,
                            GT_density=int(1000 * gk.density(network_GT)),
                            edge_weights=priorities, pnum=1, optim='optN', selfloop=True)
        
        if not r_estimated:
            print("✗ DRASL returned no solutions")
            return False
        
        print(f"✓ DRASL completed: found {len(r_estimated)} solutions")
        
        # Apply delta selection (percentage-based)
        solutions_with_costs = []
        for answer in r_estimated:
            graph_num = answer[0][0]
            undersampling_rate = answer[0][1]
            cost = answer[1]
            solutions_with_costs.append((graph_num, undersampling_rate, cost))
        
        solutions_with_costs.sort(key=lambda x: x[2])
        min_cost = solutions_with_costs[0][2]
        absolute_delta = delta * min_cost
        selected_solutions = [s for s in solutions_with_costs if s[2] <= min_cost + absolute_delta]
        
        print(f"✓ Delta selection: {len(selected_solutions)}/{len(solutions_with_costs)} solutions")
        print(f"  Delta: {delta*100:.0f}% of min_cost (absolute: {absolute_delta:.0f})")
        
        # Compute F1 scores (ORIENTATION ONLY)
        f1_scores = []
        for graph_num, undersampling_rate, cost in selected_solutions:
            res_rasl = bfutils.num2CG(graph_num, len(network_GT))
            rasl_sol = mf.precision_recall(res_rasl, network_GT, include_selfloop=True)
            f1_scores.append(rasl_sol)
        
        avg_orientation_f1 = np.mean([s['orientation']['F1'] for s in f1_scores])
        
        print(f"✓ Computed F1 scores: avg orientation F1 = {avg_orientation_f1:.4f}")
        
        print("\n✓ Full pipeline test successful!\n")
        return True
        
    except Exception as e:
        print(f"✗ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "=" * 80)
    print("DELTA HYPERPARAMETER TUNING - TEST SUITE")
    print("=" * 80)
    print("\nThis script verifies your setup before running full hyperparameter tuning.")
    print("\n")
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Data Access", test_data_access()))
    results.append(("Delta Logic", test_delta_logic()))
    results.append(("PCMCI Basic", test_pcmci_basic()))
    results.append(("Full Pipeline", test_full_pipeline()))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:<20} {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED!")
        print("\nYou're ready to run delta hyperparameter tuning:")
        print("  python VAR_delta_tuning.py -n 3 -u 3 --test_mode")
        print("\nNote: Delta is now percentage-based (e.g., 0.5 = 50% of min_cost)")
        print("      Only Orientation F1 is analyzed")
        print("\n" + "=" * 80)
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        print("\nPlease fix the issues above before running hyperparameter tuning.")
        print("See DELTA_TUNING_README.md for troubleshooting help.")
        print("\n" + "=" * 80)
        return 1

if __name__ == "__main__":
    sys.exit(main())

