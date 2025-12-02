#!/bin/bash
# Analyze results focusing ONLY on Orientation F1

set -e

echo "=========================================================================="
echo "  Hyperparameter Tuning Analysis - ORIENTATION F1 ONLY"
echo "=========================================================================="
echo ""

# Check if file provided as argument
if [ -n "$1" ]; then
    RESULTS_FILE="$1"
else
    # Auto-detect most recent results file
    echo "Searching for results file..."
    
    if ls VAR_ruben/hyperparameter_tuning/priority_tuning_*.csv 1> /dev/null 2>&1; then
        RESULTS_FILE=$(ls -t VAR_ruben/hyperparameter_tuning/priority_tuning_*.csv | head -1)
    elif ls priority_tuning_*.csv 1> /dev/null 2>&1; then
        RESULTS_FILE=$(ls -t priority_tuning_*.csv | head -1)
    else
        echo "ERROR: No results file found!"
        echo "Usage: $0 /path/to/results.csv"
        exit 1
    fi
fi

if [ ! -f "$RESULTS_FILE" ]; then
    echo "ERROR: File not found: $RESULTS_FILE"
    exit 1
fi

echo "Found: $RESULTS_FILE"
echo ""

# Get absolute path and output directory
RESULTS_FILE=$(realpath "$RESULTS_FILE")
OUTPUT_DIR=$(dirname "$RESULTS_FILE")

echo "=========================================================================="
echo "Generating plots (Orientation F1 only)..."
echo "=========================================================================="
echo ""

python VAR_analyze_orientation_only.py -f "$RESULTS_FILE" -o "$OUTPUT_DIR"

echo ""
echo "=========================================================================="
echo "Generating report (Orientation F1 only)..."
echo "=========================================================================="
echo ""

REPORT_FILE="${RESULTS_FILE%.csv}_report_ORIENTATION_ONLY.md"
python VAR_report_orientation_only.py -f "$RESULTS_FILE" -o "$REPORT_FILE"

echo ""
echo "=========================================================================="
echo "✓ COMPLETE!"
echo "=========================================================================="
echo ""
echo "Files generated in: $OUTPUT_DIR"
echo ""
echo "Plots (Orientation F1 focused):"
echo "  - orientation_f1_top.png"
echo "  - orientation_f1_distribution.png"
echo "  - priority_impact_orientation.png"
echo "  - precision_recall_orientation.png"
echo ""
echo "Data:"
echo "  - summary_statistics_orientation.csv"
echo ""
echo "Report:"
echo "  - $(basename $REPORT_FILE)"
echo ""
echo "=========================================================================="
echo "Next steps:"
echo "  1. View plots:  open $OUTPUT_DIR/*orientation*.png"
echo "  2. Read report: open $REPORT_FILE"
echo "=========================================================================="
echo ""

