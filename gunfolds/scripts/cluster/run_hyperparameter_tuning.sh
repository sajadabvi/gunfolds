#!/bin/bash
# Wrapper script for running hyperparameter tuning

set -e  # Exit on error

# Default values
NET=3
UNDERSAMPLING=3
NUM_BATCHES=5
PNUM=8
MODE="default"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MODE="test"
            shift
            ;;
        --subset)
            MODE="subset"
            shift
            ;;
        --all)
            MODE="all"
            shift
            ;;
        --net)
            NET="$2"
            shift 2
            ;;
        --undersampling)
            UNDERSAMPLING="$2"
            shift 2
            ;;
        --batches)
            NUM_BATCHES="$2"
            shift 2
            ;;
        --pnum)
            PNUM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --test              Run quick test with 2 configs (recommended first)"
            echo "  --subset            Test curated subset (~20 configs)"
            echo "  --all               Test ALL 3125 combinations (WARNING: very slow!)"
            echo "  --net N             Network number (default: 3)"
            echo "  --undersampling U   Undersampling rate (default: 3)"
            echo "  --batches B         Number of batches to average (default: 5)"
            echo "  --pnum P            Number of CPUs (default: 8)"
            echo "  --help              Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --test                           # Quick test"
            echo "  $0 --subset --net 3 --batches 5     # Subset mode"
            echo "  $0 --all --pnum 16                  # Full search with 16 CPUs"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "RASL Priority Hyperparameter Tuning"
echo "========================================================================"
echo "Configuration:"
echo "  Network: $NET"
echo "  Undersampling: $UNDERSAMPLING"
echo "  Batches: $NUM_BATCHES"
echo "  CPUs: $PNUM"
echo "  Mode: $MODE"
echo "========================================================================"
echo ""

# Run based on mode
case $MODE in
    test)
        echo "Running quick test..."
        python test_hyperparameter_tuning.py
        ;;
    default)
        echo "Running default mode (5 representative configurations)..."
        python VAR_hyperparameter_tuning.py \
            -n $NET \
            -u $UNDERSAMPLING \
            --num_batches $NUM_BATCHES \
            -p $PNUM
        ;;
    subset)
        echo "Running subset mode (~20 curated configurations)..."
        python VAR_hyperparameter_tuning.py \
            -n $NET \
            -u $UNDERSAMPLING \
            --num_batches $NUM_BATCHES \
            -p $PNUM \
            --test_subset
        ;;
    all)
        echo "WARNING: Running ALL 3125 combinations. This will take a LONG time!"
        echo "Press Ctrl+C within 5 seconds to cancel..."
        sleep 5
        echo "Starting full hyperparameter search..."
        python VAR_hyperparameter_tuning.py \
            -n $NET \
            -u $UNDERSAMPLING \
            --num_batches $NUM_BATCHES \
            -p $PNUM \
            --test_all
        ;;
esac

# If not test mode, find and analyze the most recent results
if [ "$MODE" != "test" ]; then
    echo ""
    echo "========================================================================"
    echo "Searching for results file..."
    
    # Find the most recent results file
    RESULTS_DIR="VAR_ruben/hyperparameter_tuning"
    if [ -d "$RESULTS_DIR" ]; then
        LATEST_CSV=$(ls -t ${RESULTS_DIR}/priority_tuning_net${NET}_u${UNDERSAMPLING}_*.csv 2>/dev/null | head -1)
        
        if [ -n "$LATEST_CSV" ]; then
            echo "Found: $LATEST_CSV"
            echo ""
            echo "Generating analysis and report..."
            
            # Run analysis
            python VAR_analyze_hyperparameters.py -f "$LATEST_CSV"
            
            # Generate report
            python VAR_generate_report.py -f "$LATEST_CSV"
            
            echo ""
            echo "========================================================================"
            echo "✓ Complete!"
            echo "========================================================================"
            echo "Results saved in: $RESULTS_DIR"
            echo ""
            echo "Files generated:"
            echo "  - ${LATEST_CSV}"
            echo "  - ${LATEST_CSV%.csv}_report.md"
            echo "  - ${RESULTS_DIR}/f1_comparison.png"
            echo "  - ${RESULTS_DIR}/combined_f1.png"
            echo "  - ${RESULTS_DIR}/priority_heatmap.png"
            echo "  - ${RESULTS_DIR}/precision_recall.png"
            echo "  - ${RESULTS_DIR}/summary_statistics.csv"
        else
            echo "No results file found in $RESULTS_DIR"
        fi
    else
        echo "Results directory not found: $RESULTS_DIR"
    fi
fi

echo ""
echo "Done!"

