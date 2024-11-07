import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.patches as mpatches
import argparse
import distutils.util
from gunfolds.utils import zickle as zkl
from os import listdir
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import re
import glob

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="t", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="f", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))


def sort_key(filename):
    # Use regular expression to extract the number between "_undersampled_by_" and "_batch_"
    match = re.search(r'_undersampled_by_(\d+)_batch_', filename)
    if match:
        return int(match.group(1))  # Extract the number and convert it to an integer
    return float('inf')  # Return a large number if no match (place those at the end)


methods = ['FASK', 'RASL','mRASL']
save_results = []
for snr in range(1, 6):
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

        POSTFIX = 'VAAR_BOLD_noise_ruben_nets'

        folder = f'/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/VAR_ringmore/BOLD_noise/snr{snr}/{method}/'
        items = listdir(folder)
        items.sort(key=sort_key)
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

        # Append the result along with the snr value
        save_results.append((snr, result))

# Convert the `save_results` list into a DataFrame
data_records = []

for snr, result in save_results:
    for method, method_data in result.items():
        for undersampling, metrics in method_data.items():
            for metric in ['F1_O', 'F1_A', 'F1_C']:
                if metric in metrics:
                    values = metrics[metric]
                    for value in values:
                        data_records.append({
                            'snr': snr,
                            'method': method,
                            'undersampling': undersampling,
                            'metric': metric,
                            'value': value
                        })

df = pd.DataFrame(data_records)

# Map metric names to facet titles
facet_titles = {
    'F1_O': 'Orientation',
    'F1_A': 'Adjacency',
    'F1_C': 'Cycle Detection'
}

df['facet'] = df['metric'].map(facet_titles)
df['undersampling'] = df['undersampling'].astype(str)  # Ensure 'undersampling' is a string for proper plotting
# Convert 'method' to numeric codes for plotting
df['method_code'] = pd.Categorical(df['method']).codes

# Set up the FacetGrid with 'snr' as rows
g = sns.FacetGrid(df, row='undersampling', col='facet', sharey=False, height=4, aspect=1.5)

# Map boxplots to the grid with adjusted width
g.map_dataframe(
    sns.boxplot,
    x='method',
    y='value',
    hue='snr',
    palette='Set2',
    dodge=True,
    width=0.5  # Adjusted width to make boxes slimmer
)

# Adjust the legend
g.add_legend(title='snr', title_fontsize=12, fontsize=10)

# Customize axes and titles
for ax in g.axes.flat:
    ax.grid(True)
    ax.set_ylim(0, 1)
    # Set x-ticks to method names
    ax.set_xticks(np.arange(len(df['method'].unique())))
    ax.set_xticklabels(df['method'].unique(), fontsize=10)
    # Adjust y-tick labels
    ax.tick_params(axis='y', labelsize=10)
    # Adjust x-axis limits to ensure all boxes are visible
    ax.set_xlim(-0.5, len(df['method'].unique()) - 0.5)

# Adjust titles and labels
g.set_axis_labels('Method', 'F1 Score')
g.set_titles(row_template="SNR={row_name}", col_template="{col_name}")

# Adjust layout and add an overall title
plt.subplots_adjust(top=0.9)
g.fig.suptitle('F1 Scores Across Different SNR Levels', fontsize=16)

# Save the figure
now = str(datetime.now())
now = now[:-7].replace(' ', '_')
filename = POSTFIX + '_' + now
plt.savefig(filename + '_grouped_boxplot.png')
