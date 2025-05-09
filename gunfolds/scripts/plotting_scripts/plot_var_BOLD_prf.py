import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse
import distutils.util
from gunfolds.utils import zickle as zkl
from os import listdir
from collections import defaultdict
import pandas as pd
import seaborn as sns
import re

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-u", "--UNDERSAMPLED", default="f", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))


def sort_key(filename):
    match = re.search(r'_undersampled_by_(\d+)_batch_', filename)
    return int(match.group(1)) if match else float('inf')


methods = ['MVGC', 'MVAR', 'GIMME', 'PC', 'FASK', 'RASL']
save_results = []

for method in methods:
    aggregated_data = defaultdict(lambda: defaultdict(lambda: {
        'Precision_O': [], 'Recall_O': [], 'F1_O': [],
        'Precision_A': [], 'Recall_A': [], 'F1_A': [],
        'Precision_C': [], 'Recall_C': [], 'F1_C': [],
    }))

    folder = f'/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/VAR_ringmore/BOLD/{method}/'
    items = sorted([item for item in listdir(folder) if not item.startswith('.')], key=sort_key)

    for item in items:
        data = zkl.load(folder + item)
        if method in data:
            for key in data[method]:
                if key == 1:
                    continue
                for metric_name, values in data[method][key].items():
                    aggregated_data[method][key][metric_name].extend(values)

    save_results.append({k: dict(v) for k, v in aggregated_data.items()})

data_records = []

for result in save_results:
    for method, method_data in result.items():
        for undersampling, metrics in method_data.items():
            for metric_name, values in metrics.items():
                if "_" not in metric_name:
                    continue
                metric_type, aspect = metric_name.split("_")
                for v in values:
                    data_records.append({
                        "method": method, "undersampling": str(undersampling),
                        "metric_type": metric_type, "aspect": aspect, "value": v
                    })

df = pd.DataFrame(data_records)

aspect_map = {'O': 'Orientation', 'A': 'Adjacency', 'C': 'Cycle Detection'}
metric_map = {'F1': 'F1 Score', 'Precision': 'Precision', 'Recall': 'Recall'}

df['aspect_label'] = df['aspect'].map(aspect_map)
df['metric_label'] = df['metric_type'].map(metric_map)
df['method'] = pd.Categorical(df['method'], categories=methods, ordered=True)

# ---- Create FacetGrid ---- #
g = sns.FacetGrid(df, row='metric_label', col='aspect_label', sharey=False, height=5, aspect=0.8)

g.map_dataframe(sns.boxplot, x='method', y='value', hue='undersampling', palette='Set2', dodge=True)

# ---- Formatting: Add ticks, grid, and enlarge labels ---- #
for ax in g.axes.flatten():
    ax.set_ylim(0, 1)

    # Major ticks on X and Y axes
    ax.xaxis.set_tick_params(labelsize=14, rotation=30)
    ax.yaxis.set_tick_params(labelsize=14)

    # Enable minor ticks for better readability
    # ax.minorticks_on()

    # Customize major and minor ticks
    ax.tick_params(axis='both', which='major', length=8, width=1.5)  # Major ticks
    # ax.tick_params(axis='both', which='minor', length=4, width=1, color='gray')  # Minor ticks

    # Add grid for better visualization
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set axis labels
    ax.set_xlabel("Method", fontsize=16, labelpad=12)
    ax.set_ylabel("Score", fontsize=16, labelpad=12)

# ---- Adjust Legends and Titles ---- #
g.add_legend(title='Undersampling', title_fontsize=18, fontsize=16)
g.set_titles(row_template="{row_name}", col_template="{col_name}", size=20)

plt.suptitle(
    'Precision, Recall, and F1 Scores\nVAR simulations + BOLD for random ring graphs (size 5 to 10)\nAcross different undersampling levels',
    fontsize=22
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save and show plot
now = str(datetime.now())[:-7].replace(' ', '_')
filename = 'VAAR_BOLD_ruben_nets' + '_' + now
plt.savefig(filename + '_bold_prf_v2.svg')
# plt.show()
