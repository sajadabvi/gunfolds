import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from gunfolds.utils import zickle as zkl

# -------------------------
# Reported values for FASK and Alasso
# (If you only have FASK/Alasso for certain metrics, adjust accordingly.)
reported_values = {
    'F1': {
        'FASK': 0.425,   # from your earlier calculations
        'Alasso': 0.9035
    },
    'Precision': {
        'FASK': 0.95,
        'Alasso': 0.97
    },
    'Recall': {
        'FASK': 0.275,   # average of 0.24 and 0.31
        'Alasso': 0.845  # average of 0.84 and 0.85
    }
}

# -------------------------
# 1. Load real data (RASL) from your .zkl files
results_folder = 'macaque_results/RASL'
files = sorted([
    f for f in os.listdir(results_folder)
    if f.endswith('.zkl') and 'RASL' in f  # only load RASL files
])

rasl_records = []
metrics_required = ['F1_O','F1_A','Precision_O','Precision_A','Recall_O','Recall_A']

for filename in files:
    filepath = os.path.join(results_folder, filename)
    data = zkl.load(filepath)
    # The data structure is { 'RASL': { numeric_key: { ...metrics... } } }
    # We'll combine F1_O/F1_A etc. to get a distribution for RASL.
    if 'RASL' not in data:
        continue
    for numeric_key, subdict in data['RASL'].items():
        if all(k in subdict for k in metrics_required):
            # We'll iterate over the min length in case lists differ in size
            count = min(len(subdict['F1_O']), len(subdict['F1_A']),
                        len(subdict['Precision_O']), len(subdict['Precision_A']),
                        len(subdict['Recall_O']), len(subdict['Recall_A']))
            for i in range(count):
                # F1
                f1_val = (subdict['F1_O'][i] + subdict['F1_A'][i]) / 2.0
                rasl_records.append({
                    'method': 'RASL',
                    'metric': 'F1',
                    'value': f1_val
                })
                # Precision
                prec_val = (subdict['Precision_O'][i] + subdict['Precision_A'][i]) / 2.0
                rasl_records.append({
                    'method': 'RASL',
                    'metric': 'Precision',
                    'value': prec_val
                })
                # Recall
                recall_val = (subdict['Recall_O'][i] + subdict['Recall_A'][i]) / 2.0
                rasl_records.append({
                    'method': 'RASL',
                    'metric': 'Recall',
                    'value': recall_val
                })

# -------------------------
# 2. Generate synthetic data for FASK and Alasso
#    We'll create 10 random points for each metric, for each method,
#    within ±5% of the reported mean.

fask_alasso_records = []
num_points = 10
rng = np.random.default_rng(seed=42)  # reproducible random numbers

for metric in ['F1', 'Precision', 'Recall']:
    for method in ['FASK', 'Alasso']:
        if method not in reported_values[metric]:
            # If no reported value is given, skip
            continue
        mean_val = reported_values[metric][method]
        delta = 0.05 * mean_val  # ±5% of the reported mean
        synthetic_vals = rng.uniform(mean_val - delta, mean_val + delta, size=num_points)
        for val in synthetic_vals:
            fask_alasso_records.append({
                'method': method,
                'metric': metric,
                'value': val
            })

# -------------------------
# Combine RASL records with FASK/Alasso synthetic data
all_records = rasl_records + fask_alasso_records
df = pd.DataFrame(all_records)

print("DataFrame head:")
print(df.head())

# -------------------------
# 3. Plot: We want three facets (F1, Precision, Recall), with 3 methods on each x-axis.

# Make sure the method order is [FASK, Alasso, RASL]
method_order = ['FASK', 'Alasso', 'RASL']

g = sns.FacetGrid(df, col='metric', sharey=False, height=5, aspect=0.7)
# Use method as x and no hue to keep the box color consistent.
# We'll fix the palette by specifying color or a custom palette argument.
# However, to avoid the palette-without-hue warning, we can just set a default color or pass a dummy hue.

def facet_boxplot(data, **kwargs):
    sns.boxplot(
        data=data,
        x='method',
        y='value',
        order=method_order,
        color='#82c6a4',  # a green-ish color
        width=0.5
    )
    # overlay stripplot
    sns.stripplot(
        data=data,
        x='method',
        y='value',
        order=method_order,
        color='black',
        size=4,
        jitter=True,
        linewidth=0.4
    )

g.map_dataframe(facet_boxplot)

for ax in g.axes.flat:
    # ax.set_ylim(0, 1)
    ax.grid(True, color='#d0e4f5', linewidth=1.0, linestyle='-')  # light blue grid

# -------------------------
# 4. Overlay the reported means as red diamond markers (only for FASK, Alasso).
for ax, metric in zip(g.axes.flat, g.col_names):
    # The x-ticks are the 3 methods in method_order
    for i, method in enumerate(method_order):
        # RASL has no "reported" average, so skip it
        if method in ['FASK', 'Alasso']:
            rep_val = reported_values[metric][method]
            ax.plot(
                i, rep_val,
                marker='D', markersize=8, color='red',
                label='Reported Value' if i == 0 else ""
            )
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    # Add a custom legend entry for reported value if not present
    handles, labels = ax.get_legend_handles_labels()
    if 'Reported Value' not in labels:
        red_patch = mpatches.Patch(color='red', label='Reported Value')
        handles.append(red_patch)
        labels.append('Reported Value')
    ax.legend(handles=handles, labels=labels, loc='best')

g.set_titles(col_template="{col_name}", size=14)
plt.suptitle("Boxplots of F1, Precision, Recall for FASK, Alasso, RASL\n(±5% synthetic data for FASK & Alasso)",
             fontsize=16, y=1.05)
plt.tight_layout()

# -------------------------
# 5. Save and show
now = str(datetime.now())[:-7].replace(' ', '_')
plt.savefig(f'Metrics_plot_{now}.png', dpi=300)
plt.show()
