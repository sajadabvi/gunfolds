#!/bin/bash
# Quick script to analyze hyperparameter tuning results

set -e

echo "=========================================================================="
echo "  Hyperparameter Tuning Results - Analysis & Visualization"
echo "=========================================================================="
echo ""

# Check if file provided as argument
if [ -n "$1" ]; then
    RESULTS_FILE="$1"
else
    # Auto-detect most recent results file
    echo "Searching for results file..."
    
    # Try different locations
    if ls VAR_ruben/hyperparameter_tuning/priority_tuning_*.csv 1> /dev/null 2>&1; then
        RESULTS_FILE=$(ls -t VAR_ruben/hyperparameter_tuning/priority_tuning_*.csv | head -1)
    elif ls priority_tuning_*.csv 1> /dev/null 2>&1; then
        RESULTS_FILE=$(ls -t priority_tuning_*.csv | head -1)
    elif ls *.csv 1> /dev/null 2>&1; then
        # Look for any CSV with "priority" in the name
        RESULTS_FILE=$(ls -t *priority*.csv 2>/dev/null | head -1)
    else
        echo ""
        echo "ERROR: No results file found!"
        echo ""
        echo "Usage:"
        echo "  $0 /path/to/results.csv"
        echo ""
        echo "Or place results.csv in current directory or VAR_ruben/hyperparameter_tuning/"
        exit 1
    fi
fi

# Check if file exists
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
echo "Step 1: Generating plots and statistics..."
echo "=========================================================================="
echo ""

python VAR_analyze_hyperparameters.py -f "$RESULTS_FILE" -o "$OUTPUT_DIR"

echo ""
echo "=========================================================================="
echo "Step 2: Generating comprehensive report..."
echo "=========================================================================="
echo ""

REPORT_FILE="${RESULTS_FILE%.csv}_report.md"
python VAR_generate_report.py -f "$RESULTS_FILE" -o "$REPORT_FILE"

echo ""
echo "=========================================================================="
echo "✓ COMPLETE!"
echo "=========================================================================="
echo ""
echo "Files generated in: $OUTPUT_DIR"
echo ""
echo "Plots:"
echo "  - f1_comparison.png"
echo "  - combined_f1.png"
echo "  - priority_heatmap.png"
echo "  - precision_recall.png"
echo ""
echo "Data:"
echo "  - summary_statistics.csv"
echo "  - ${RESULTS_FILE##*/}"
echo ""
echo "Report:"
echo "  - ${REPORT_FILE##*/}"
echo ""
echo "=========================================================================="
echo "Next steps:"
echo "  1. View plots:  open $OUTPUT_DIR/*.png"
echo "  2. Read report: open $REPORT_FILE"
echo "  3. Check stats: open $OUTPUT_DIR/summary_statistics.csv"
echo "=========================================================================="
echo ""

