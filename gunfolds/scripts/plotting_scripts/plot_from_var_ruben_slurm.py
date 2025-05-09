import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import argparse
import distutils.util
from gunfolds.utils import zickle  as zkl
from os import listdir
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import glob

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="t", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="f", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))

methods = ['MVGC', 'MVAR', 'GIMME', 'PC','FASK', 'RASL']
save_results = []
for method in methods:
    # Initialize a defaultdict of lists to hold concatenated results for each method
    aggregated_data = defaultdict(lambda: defaultdict(lambda: {
        'Precision_O': [],
        'Recall_O': [],
        'F1_O': [],
        'Precision_A': [],
        'Recall_A': [],
        'F1_A': [],
        'Precision_C': [],
        'Recall_C': [],
        'F1_C': [],
    }))

    POSTFIX = 'VAAR_ruben_nets'


    folder = f'/Users/mabavisani/Code_local/mygit/gunfolds/gunfolds/scripts/VAR_ruben/varuben/{method}/'
    items = listdir(folder)
    items.sort()

    # Remove any files that start with '.'
    items = [item for item in items if not item.startswith('.')]

    for item in items:
        data = zkl.load(folder + item)  # Load data from file

        if method in data:
            for key in data[method]:
                if not key == 1:
                    # Access the nested dictionary for each key
                    metrics = data[method][key]

                    # Iterate over all metrics and concatenate values
                    for metric in metrics:
                        for sub_metric in metrics[metric]:
                            aggregated_data[method][key][metric].extend([sub_metric])

    # Convert defaultdict to a regular dict for the final output
    result = {k: dict(v) for k, v in aggregated_data.items()}

    # Optionally, save results or perform further operations
    save_results.append(result)

# Convert the `save_results` list into a DataFrame
data_records = []

for result in save_results:
    for method, method_data in result.items():
        for undersampling, metrics in method_data.items():
            for metric in ['F1_O', 'F1_A', 'F1_C']:
                if metric in metrics:
                    values = metrics[metric]
                    for value in values:
                        data_records.append({'method': method, 'undersampling': undersampling, 'metric': metric, 'value': value})

df = pd.DataFrame(data_records)

# Map metric names to facet titles
facet_titles = {
    'F1_O': 'Orientation',
    'F1_A': 'Adjacency',
    'F1_C': '2-Cycle'
}

df['facet'] = df['metric'].map(facet_titles)
df['undersampling'] = df['undersampling'].astype(str)  # Ensure 'undersampling' is a string for proper plotting
# Convert 'method' to numeric codes for plotting
df['method_code'] = pd.Categorical(df['method']).codes
# Set up the FacetGrid
g = sns.FacetGrid(df, col='facet', col_wrap=3, sharey=False, height=6)

# Map boxplots to the grid
g.map_dataframe(sns.boxplot, x='method', y='value', hue='undersampling', palette='Set2', dodge=True)

# Overlay individual data points
def add_jittered_points(ax, data, x_col, y_col, hue_col):
    palette = sns.color_palette('Set2')
    hue_values = data[hue_col].unique()
    for i, hue_value in enumerate(hue_values):
        subset = data[data[hue_col] == hue_value]
        # Get method codes and add jitter
        method_codes = subset['method_code'].unique()
        for method_code in method_codes:
            subset_method = subset[subset['method_code'] == method_code]
            # Adjust x-coordinate based on undersampling
            jittered_x = []
            for _, row in subset_method.iterrows():
                if row['undersampling'] == '1':
                    jittered_x.append(method_code - 0.27)
                elif row['undersampling'] == '2':
                    jittered_x.append(method_code)
                elif row['undersampling'] == '3':
                    jittered_x.append(method_code + 0.27)
            jittered_x = np.array(jittered_x) + np.random.uniform(-0.1, 0.1, size=len(jittered_x))
            ax.scatter(jittered_x, subset_method[y_col],
                       color=palette[i],
                       edgecolor='black',  # Add a black border around the dots
                       alpha=0.7,  # Slightly adjust alpha for better visibility
                       s=25,  # Size of the dots
                       linewidth=0.4,  # Thickness of the border
                       label=hue_value)

# Add grid lines to each subplot
for ax in g.axes.flat:
    ax.grid(True)
    ax.set_ylim(0, 1)
    # Add jittered points
    # add_jittered_points(ax, df, 'method_code', 'value', 'undersampling')

    # Set x-ticks to method names
    ax.set_xticks(np.arange(len(df['method'].unique())))
    ax.set_xticklabels(df['method'].unique())

# Adjust titles and labels
g.set_axis_labels('Method', 'F1 Score')
g.set_titles(col_template="{col_name}")
g.add_legend(title='Undersampling')

# Add an overall title
plt.suptitle('F1 score of VAR simulations of simple networks in Ruben data through different levels of undersampling')
plt.tight_layout(rect=[0, 0, 0.92, 0.98])


now = str(datetime.now())
now = now[:-7].replace(' ', '_')
filename = POSTFIX + '_' + now
plt.savefig(filename + '_pc_added_grouped_boxplot.svg')
# plt.show()

