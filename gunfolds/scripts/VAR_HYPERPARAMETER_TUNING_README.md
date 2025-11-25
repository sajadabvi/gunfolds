# VAR Hyperparameter Tuning for RASL Priorities

This set of scripts performs hyperparameter tuning for the RASL priority parameters used in causal discovery on VAR simulations.

## Overview

The RASL algorithm uses 5 priority values to weight different constraints during optimization. Each priority can be a value from 1-5, where:
- Higher numbers = higher priority
- Same numbers = equal priority
- Different combinations can significantly affect performance

## Files

1. **VAR_hyperparameter_tuning.py** - Main script to run hyperparameter search
2. **VAR_analyze_hyperparameters.py** - Script to analyze and visualize results
3. **VAR_for_ruben_nets.py** - Base script (reference implementation)

## Usage

### 1. Running Hyperparameter Tuning

#### Basic Usage (Default: 5 representative priority combinations)
```bash
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5
```

#### Test a Curated Subset (~20 combinations)
```bash
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_subset
```

#### Test ALL Combinations (3125 total - will take a long time!)
```bash
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_all
```

#### Command Line Arguments
- `-n, --NET`: Network number (default: 3)
- `-u, --UNDERSAMPLING`: Undersampling rate (default: 3)
- `--num_batches`: Number of batches to average over (default: 5)
- `-p, --PNUM`: Number of CPUs to use
- `--test_subset`: Test curated subset of ~20 combinations
- `--test_all`: Test all 3125 possible combinations (⚠️ very time consuming!)

### 2. Analyzing Results

After running hyperparameter tuning, analyze the results:

```bash
python VAR_analyze_hyperparameters.py -f VAR_ruben/hyperparameter_tuning/priority_tuning_net3_u3_TIMESTAMP.csv
```

This will generate:
- **f1_comparison.png** - Bar chart comparing F1 scores across metrics
- **combined_f1.png** - Combined F1 scores for top configurations
- **priority_heatmap.png** - Heatmap showing impact of each priority position/value
- **precision_recall.png** - Precision-recall scatter plots
- **summary_statistics.csv** - Statistical summary of all metrics

## Output

### Results Files

Results are saved in `VAR_ruben/hyperparameter_tuning/`:
- **CSV file**: Human-readable results table
- **ZKL file**: Full results including configuration

### Results Table Columns

- `priorities`: The tested priority configuration [p1, p2, p3, p4, p5]
- `p1` to `p5`: Individual priority values
- `num_successful_batches`: How many batches completed successfully
- `orientation_precision`, `orientation_recall`, `orientation_F1`: Metrics for edge orientation
- `adjacency_precision`, `adjacency_recall`, `adjacency_F1`: Metrics for edge adjacency
- `cycle_precision`, `cycle_recall`, `cycle_F1`: Metrics for 2-cycles
- `combined_F1`: Average of all three F1 scores (overall metric)

## Understanding Priority Values

The 5 priorities correspond to different constraints in the RASL optimization:
1. **p1**: Constraint 1 priority
2. **p2**: Constraint 2 priority
3. **p3**: Constraint 3 priority
4. **p4**: Constraint 4 priority
5. **p5**: Constraint 5 priority

### Example Configurations

- `[4, 2, 5, 3, 1]` - Original configuration (mixed priorities)
- `[1, 1, 1, 1, 1]` - All constraints equal priority
- `[5, 4, 3, 2, 1]` - Descending priorities
- `[5, 5, 5, 5, 5]` - All high priority

## Recommendations

1. **Start with default** - Run default mode first (5 configurations) to test the pipeline
2. **Try subset mode** - Use `--test_subset` for ~20 curated combinations
3. **Full search** - Only use `--test_all` if you have significant computational resources and time

## Example Workflow

```bash
# 1. Run hyperparameter tuning (subset mode)
python VAR_hyperparameter_tuning.py -n 3 -u 3 --num_batches 5 --test_subset -p 8

# 2. Analyze results
python VAR_analyze_hyperparameters.py -f VAR_ruben/hyperparameter_tuning/priority_tuning_net3_u3_20250125_143022.csv

# 3. Review top configurations and select best
# Results are sorted by combined_F1 score
```

## Notes

- Each priority combination is tested on batches 1 through `num_batches`
- Results are averaged across successful batches
- The `combined_F1` metric is the average of orientation_F1, adjacency_F1, and cycle_F1
- Higher F1 scores are better (range: 0.0 to 1.0)

## Requirements

- Data files must exist in: `~/DataSets_Feedbacks/8_VAR_simulation/net{NET}/u{UNDERSAMPLING}/txtSTD/`
- tqdm package for progress bars
- Standard gunfolds dependencies

