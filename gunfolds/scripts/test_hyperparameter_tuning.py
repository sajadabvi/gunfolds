"""
Quick test script to verify hyperparameter tuning setup
This runs just 2 priority configurations on 2 batches for quick testing
"""
import os
import sys
from VAR_hyperparameter_tuning import (
    parse_arguments, 
    run_batch, 
    average_metrics,
    get_priority_combinations
)
from gunfolds.scripts.datasets.simple_networks import simp_nets

def test_setup():
    """Test that all required files and imports work"""
    print("Testing setup...")
    
    # Test imports
    try:
        import numpy as np
        import pandas as pd
        from gunfolds import conversions as cv
        from gunfolds.solvers.clingo_rasl import drasl
        from gunfolds.utils import graphkit as gk
        from gunfolds.utils import bfutils
        from gunfolds.utils import zickle as zkl
        import tigramite.data_processing as pp
        from tigramite.pcmci import PCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test data file exists
    net = 3
    undersampling = 3
    batch = 1
    path = os.path.expanduser(
        f'~/DataSets_Feedbacks/8_VAR_simulation/net{net}/u{undersampling}/txtSTD/data{batch}.txt')
    
    if os.path.exists(path):
        print(f"✓ Data file found: {path}")
    else:
        print(f"✗ Data file not found: {path}")
        print("  Please ensure data files exist before running hyperparameter tuning")
        return False
    
    # Test network loading
    try:
        network_GT = simp_nets(net, selfloop=True)
        print(f"✓ Network {net} loaded: {len(network_GT)} nodes")
    except Exception as e:
        print(f"✗ Error loading network: {e}")
        return False
    
    return True

def quick_test():
    """Run a quick test with 2 priority sets and 2 batches"""
    print("\n" + "=" * 80)
    print("QUICK TEST: 2 priority configurations × 2 batches")
    print("=" * 80)
    
    # Simple test configuration
    class TestArgs:
        NET = 3
        UNDERSAMPLING = 3
        MAXU = 5
        PNUM = 2  # Use fewer CPUs for testing
        num_batches = 2
    
    args = TestArgs()
    
    # Set environment
    os.environ['OMP_NUM_THREADS'] = str(args.PNUM)
    
    # Load network
    network_GT = simp_nets(args.NET, selfloop=True)
    
    # Test just 2 priority combinations
    test_priorities = [
        [4, 2, 5, 3, 1],  # Original
        [1, 1, 1, 1, 1],  # All equal
    ]
    
    results = []
    
    for priorities in test_priorities:
        print(f"\nTesting priorities: {priorities}")
        
        batch_metrics = []
        for batch_num in range(1, 3):  # Just batches 1 and 2
            print(f"  Batch {batch_num}...", end=" ")
            try:
                metrics = run_batch(args, network_GT, list(priorities), batch_num)
                if metrics:
                    batch_metrics.append(metrics)
                    print("✓")
                else:
                    print("✗ (returned None)")
            except Exception as e:
                print(f"✗ Error: {e}")
        
        if batch_metrics:
            avg_metrics = average_metrics(batch_metrics)
            if avg_metrics:
                combined_f1 = (avg_metrics['orientation']['F1'] + 
                              avg_metrics['adjacency']['F1'] + 
                              avg_metrics['cycle']['F1']) / 3.0
                results.append({
                    'priorities': priorities,
                    'combined_F1': combined_f1,
                    'successful_batches': len(batch_metrics)
                })
                print(f"  ✓ Combined F1: {combined_f1:.4f}")
    
    if results:
        print("\n" + "=" * 80)
        print("TEST RESULTS:")
        print("=" * 80)
        for r in results:
            print(f"Priorities {r['priorities']}: F1={r['combined_F1']:.4f} ({r['successful_batches']}/2 batches)")
        print("\n✓ Quick test completed successfully!")
        print("  You can now run the full hyperparameter tuning script.")
        return True
    else:
        print("\n✗ Test failed - no successful results")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("HYPERPARAMETER TUNING - SETUP TEST")
    print("=" * 80)
    
    # Test setup
    if not test_setup():
        print("\n✗ Setup test failed. Please fix the issues above.")
        sys.exit(1)
    
    # Quick functional test
    print("\n" + "=" * 80)
    print("Running quick functional test...")
    print("=" * 80)
    
    try:
        if quick_test():
            print("\n" + "=" * 80)
            print("✓ ALL TESTS PASSED")
            print("=" * 80)
            print("\nYou can now run:")
            print("  python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5")
            sys.exit(0)
        else:
            print("\n✗ TESTS FAILED")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

