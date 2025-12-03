import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime
from gunfolds.utils import zickle as zkl

# File paths
file_paths = [
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/RASL_fig4.zkl",
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/FASK_fig4.zkl",
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/PCMCI_fig4.zkl",
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/GIMME_fig4.zkl",
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/MVAR_fig4.zkl",
    "/Users/mabavisani/code_local/mygit/gunfolds/gunfolds/scripts/MVGC_fig4.zkl"
]

methods = ['RASL', 'FASK', 'PCMCI', 'GIMME', 'MVAR', 'MVGC']
data_records = []

# Load and process data
for method, file_path in zip(methods, file_paths):
    data = zkl.load(file_path)  # Load the .zkl file

    # Extract precision, recall, and F1 scores for each metric
    for metric_name, metric_values in zip(['Precision', 'Recall', 'F1'], data):
        for sub_metric_name, values in zip(['O', 'A', 'C'], metric_values):
            for value in values:
                data_records.append({
                    'method': method,
                    'metric': f"{metric_name}_{sub_metric_name}",
                    'value': value
                })

# Convert the list of records to a DataFrame
df = pd.DataFrame(data_records)

# Flatten the 'value' column
df['value'] = df['value'].apply(lambda x: x if isinstance(x, list) else [x])
df = df.explode('value')

# Convert 'value' to numeric
df['value'] = pd.to_numeric(df['value'], errors='coerce')

# Drop any rows with NaN values in 'value'
df = df.dropna(subset=['value'])

# Map metric names to facet titles
facet_titles = {
    'F1_O': 'Orientation',
    'F1_A': 'Adjacency',
    'F1_C': 'Cycle Detection'
}
df['facet'] = df['metric'].map(facet_titles)

# Set up the FacetGrid
g = sns.FacetGrid(df, col='facet', col_wrap=3, sharey=False, height=6)

# Map boxplots to the grid
g.map_dataframe(sns.boxplot, x='method', y='value', palette='Set2')

# Overlay individual data points
def add_jittered_points(ax, data, x_col, y_col):
    """Add jittered points to the boxplots."""
    unique_methods = data[x_col].unique()
    for i, method in enumerate(unique_methods):
        subset = data[data[x_col] == method]
        x_coords = np.random.normal(loc=i, scale=0.1, size=len(subset))  # Jittered x-coordinates
        ax.scatter(x_coords, subset[y_col],
                   color='black',  # Color of the dots
                   alpha=0.8,  # Slightly transparent
                   s=30,  # Size of the dots
                   edgecolor='white',  # White border around the dots
                   linewidth=0.5,
                   zorder=3)  # Set z-order higher than boxplots

# Add grid lines, individual points, and adjust x-ticks
for ax, facet_data in zip(g.axes.flat, g.facet_data()):
    facet_df = facet_data[1]
    add_jittered_points(ax, facet_df, 'method', 'value')
    ax.grid(True, zorder=1)  # Keep grid lines in the background
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(df['method'].unique())))
    ax.set_xticklabels(df['method'].unique(), rotation=45, fontsize=10)

# Adjust titles and labels
g.set_axis_labels('Method', 'Value', fontsize=14)
g.set_titles(col_template="{col_name}", size=14)
plt.suptitle('Metric Scores by Method', fontsize=16)
plt.tight_layout(rect=[0, 0, 0.92, 0.98])

# Save the plot
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
filename = f"metrics_plot_{now}.png"
plt.savefig(filename)

# Show the plot
plt.show()