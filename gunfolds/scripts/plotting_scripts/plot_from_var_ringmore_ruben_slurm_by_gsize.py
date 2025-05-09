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
import glob
import re


def infer_graph_size(file_name):
    # Use regular expression to extract the batch number
    match = re.search(r'_batch_(\d+)\.zkl', file_name)
    if match:
        batch_number = int(match.group(1))

        # Check if the file name contains "_mRASL_"
        if "_mRASL_" in file_name:
            # For "_mRASL_", batches range from 1 to 60, with different size logic
            if 1 <= batch_number <= 10:
                return 5
            elif 11 <= batch_number <= 20:
                return 6
            elif 21 <= batch_number <= 30:
                return 7
            elif 31 <= batch_number <= 40:
                return 8
            elif 41 <= batch_number <= 50:
                return 9
            elif 51 <= batch_number <= 60:
                return 10
            else:
                return "Invalid batch number for mRASL"
        else:
            # For files without "_mRASL_", apply the original batch-size logic
            if 1 <= batch_number <= 60:
                return 5
            elif 61 <= batch_number <= 120:
                return 6
            elif 121 <= batch_number <= 180:
                return 7
            elif 181 <= batch_number <= 240:
                return 8
            elif 241 <= batch_number <= 300:
                return 9
            elif 301 <= batch_number <= 360:
                return 10
            else:
                return "Invalid batch number"
    else:
        return "Invalid file name format"


# Parse arguments
parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-p", "--PNUM", default=4, help="number of CPUs in machine.", type=int)
parser.add_argument("-c", "--CONCAT", default="t", help="true to use concat data", type=str)
parser.add_argument("-u", "--UNDERSAMPLED", default="f", help="true to use tr 3 time scale", type=str)
args = parser.parse_args()
PNUM = args.PNUM
UNDERSAMPLED = bool(distutils.util.strtobool(args.UNDERSAMPLED))

methods = ['MVGC', 'MVAR', 'GIMME', 'FASK', 'RASL', 'mRASL']
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

    POSTFIX = 'VAAR_ringmore_V_ruben'

    folder = f'/Users/sajad/Code_local/mygit/gunfolds/gunfolds/scripts/VAR_ringmore/VAR_ringmore/{method}/'
    items = listdir(folder)
    items.sort()

    # Remove any files that start with '.'
    items = [item for item in items if not item.startswith('.')]

    for item in items:
        data = zkl.load(folder + item)  # Load data from file
        graph_size = infer_graph_size(item)  # Get the graph size

        if method in data:
            for key in data[method]:
                if not key == 1:
                    # Access the nested dictionary for each key
                    metrics = data[method][key]

                    # Iterate over all metrics and concatenate values
                    for metric in metrics:
                        for sub_metric in metrics[metric]:
                            # Append a tuple of (sub_metric, graph_size) instead of just sub_metric
                            aggregated_data[method][key][metric].extend([(sub_metric, graph_size)])

    # Convert defaultdict to a regular dict for the final output
    result = {k: dict(v) for k, v in aggregated_data.items()}

    # Optionally, save results or perform further operations
    save_results.append(result)

# Convert the `save_results` list into a DataFrame
data_records = []

for result in save_results:  # Only iterate over the results
    for method, method_data in result.items():
        for undersampling, metrics in method_data.items():
            for metric in ['F1_O', 'F1_A', 'F1_C']:
                if metric in metrics:
                    values = metrics[metric]
                    # Now each value is a tuple (sub_metric, graph_size)
                    for sub_metric, graph_size in values:  # Unpack the tuple here
                        data_records.append({
                            'method': method,
                            'undersampling': undersampling,
                            'metric': metric,
                            'value': sub_metric,  # sub_metric is the actual metric value
                            'graph_size': graph_size  # Add graph_size to data
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
df['graph_size'] = df['graph_size'].astype(str)  # Convert graph_size to string for FacetGrid rows
# Convert 'method' to numeric codes for plotting
df['method_code'] = pd.Categorical(df['method']).codes

# Set up the FacetGrid with both row and column facets (metrics and graph size)
g = sns.FacetGrid(df, col='facet', row='graph_size', sharey=False, height=4, aspect=1.5)

# Map boxplots to the grid
g.map_dataframe(sns.boxplot, x='method', y='value', hue='undersampling', palette='Set2', dodge=True)


# Function to overlay data point counts on the boxes
def add_data_counts(ax, data, x_col, hue_col, y_col):
    # Calculate the count for each box using groupby on all three columns
    grouped = data.groupby([x_col, hue_col, y_col]).size().reset_index(name='counts')

    # Calculate means for proper placement of text
    means = data.groupby([x_col, hue_col, y_col])['value'].mean().reset_index(name='mean_value')

    # Merge the counts and means dataframes
    merged = pd.merge(grouped, means, on=[x_col, hue_col, y_col])

    # Iterate over unique combinations of x and hue
    unique_x_hue = merged.groupby([x_col, hue_col])

    for (x_val, hue_val), group in unique_x_hue:
        count = group['counts'].values[0]  # Get the count from the first row of the group
        mean = group['mean_value'].values[0]  # Get the mean from the first row of the group

        # Adjust the x coordinate for the hue offset (since dodge=True in boxplot)
        offset = {'1': -0.27, '2': 0, '3': 0.27}[str(hue_val)]

        # Place text at the mean value, for only one time per box
        ax.text(x_val + offset, mean + 0.02, f'{count}', ha='center', va='bottom', color='black', fontsize=10)


# Apply the function to each axis in the FacetGrid
for ax in g.axes.flat:
    ax.grid(True)
    ax.set_ylim(0, 1)
    # grouped = df.groupby(['method_code', 'undersampling', 'graph_size']).size().reset_index(name='counts')
    # means = df.groupby(['method_code', 'undersampling', 'graph_size'])['value'].mean().reset_index(name='mean_value')
    # merged = pd.merge(grouped, means, on=['method_code', 'undersampling', 'graph_size'])
    # unique_x_y_hue = merged.groupby(['method_code', 'undersampling', 'graph_size'])
    # for (x_val, hue_val, y_val), group in unique_x_y_hue:
    #     count = group['counts'].values[0]
    #     mean = group['mean_value'].values[0]
    #     ax.text(x_val , mean + 0.02, f'{count}', ha='center', va='bottom', color='black', fontsize=10)

    # add_data_counts(ax, df, 'method_code', 'undersampling', 'graph_size')

    # Set x-ticks to method names
    ax.set_xticks(np.arange(len(df['method'].unique())))
    ax.set_xticklabels(df['method'].unique(), fontsize=12)  # Increase x-tick label size

# Adjust titles and labels
g.set_axis_labels('Method', 'F1 Score', fontsize=17)  # Increase axis label size
g.set_titles(col_template="{col_name} | Graph Size: {row_name}", size=16)  # Increase facet title size
g.add_legend(title='Undersampling', title_fontsize=22, fontsize=17)

# Add an overall title
plt.suptitle(
    'F1 score of VAR simulations of random ring graphs of size 5 to 10 through different levels of undersampling',
    fontsize=18)
plt.tight_layout(rect=[0, 0, 0.92, 0.98])

# Save the plot
now = str(datetime.now())
now = now[:-7].replace(' ', '_')
filename = POSTFIX + '_' + now
plt.savefig(filename + '_grouped_boxplot.png')
