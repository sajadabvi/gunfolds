"""
Generate compact, high-visibility manuscript figures for GCM and RASL results.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

# --- Data Loading Functions ---

def load_gcm_results(timestamp):
    """Load GCM results from timestamp directory"""
    base_dir = f"gcm_roebroeck/{timestamp}"
    rate_file = f"{base_dir}/csv/group_edge_rate.csv"
    if not os.path.exists(rate_file):
        raise FileNotFoundError(f"Results not found: {rate_file}")
    edge_rate = pd.read_csv(rate_file, index_col=0)
    return edge_rate

def load_fmri_results(timestamp, group='combined'):
    """Load fMRI RASL results from timestamp directory"""
    base_dir = f"fbirn_results/{timestamp}"
    csv_file = f"{base_dir}/combined/group_edge_counts_combined.csv"
    summary_file = f"{base_dir}/combined/analysis_summary.csv"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results not found: {csv_file}")
    
    edge_df = pd.read_csv(csv_file)
    nodes = sorted(list(set(edge_df['src'].unique()) | set(edge_df['dst'].unique())))
    n_nodes = len(nodes)
    edge_matrix = np.zeros((n_nodes, n_nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for _, row in edge_df.iterrows():
        i = node_to_idx[row['src']]
        j = node_to_idx[row['dst']]
        edge_matrix[i, j] = row['count']
    
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        n_solutions = summary_df['Num_Solutions'].sum()
    else:
        n_solutions = edge_matrix.max()
    
    edge_rate = edge_matrix / n_solutions
    edge_rate_df = pd.DataFrame(edge_rate, index=nodes, columns=nodes)
    return edge_rate_df

# --- Plotting Function ---

def plot_network_compact(edge_rate, threshold=0.3, output_path=None, title=None):
    """
    Create a compact circular network plot with large fonts.
    """
    # Standardize node order for consistency across plots
    # Order: CEN (rPPC, rDLPFC), Salience (rFIC, ACC), DMN (PCC, VMPFC)
    standard_order = ['rPPC', 'rDLPFC', 'rFIC', 'ACC', 'PCC', 'VMPFC']
    
    # Verify and reindex
    available_nodes = [n for n in standard_order if n in edge_rate.columns]
    missing_nodes = [n for n in standard_order if n not in edge_rate.columns]
    
    # Append any extra nodes that weren't in standard list (just in case)
    extra_nodes = [n for n in edge_rate.columns if n not in standard_order]
    final_order = available_nodes + extra_nodes
    
    if missing_nodes:
        print(f"Warning: Missing nodes from standard order: {missing_nodes}")
        
    edge_rate = edge_rate.reindex(index=final_order, columns=final_order)
    node_names = list(edge_rate.columns)
    N = len(node_names)
    
    # Network colors
    network_colors = {
        'rPPC': '#E64B35',    # CEN - Red
        'rDLPFC': '#E64B35',  # CEN - Red
        'rFIC': '#4DBBD5',    # Salience - Blue
        'ACC': '#4DBBD5',     # Salience - Blue
        'PCC': '#00A087',     # DMN - Green
        'VMPFC': '#00A087'    # DMN - Green
    }
    
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(i, label=name, color=network_colors.get(name, '#666666'))
    
    edge_matrix = edge_rate.to_numpy()
    for i in range(N):
        for j in range(N):
            if i != j and edge_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=edge_matrix[i, j])
    
    # Compact layout
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, N, endpoint=False)
    pos = {i: (np.cos(angles[i])*0.85, np.sin(angles[i])*0.85) for i in range(N)} # Pull nodes in slightly
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(8, 8), facecolor='white')
    
    node_colors = [G.nodes[i]['color'] for i in range(N)]
    
    # Draw nodes (Larger relative to figure)
    nx.draw_networkx_nodes(G, pos, node_size=4500, node_color=node_colors,
                           edgecolors='white', linewidths=3, ax=ax)
    
    # Draw labels (Significantly larger font)
    nx.draw_networkx_labels(G, pos, 
                           labels={i: node_names[i] for i in range(N)},
                           font_size=18, font_weight='bold', ax=ax) # Increased from 14 to 18+
    
    # Edge properties
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    if len(weights) > 0:
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            norm_weights = (weights - w_min) / (w_max - w_min)
        else:
            norm_weights = np.ones(len(weights))
        
        widths = 1.5 + norm_weights * 8  # Thicker base width
        alphas = 0.5 + norm_weights * 0.5
        
        for (u, v), width, alpha in zip(G.edges(), widths, alphas):
            has_reverse = G.has_edge(v, u)
            edge_color = '#333333'
            arrow_size = 35 # Larger arrows
            
            if has_reverse and u < v:
                rad = 0.3
                nx.draw_networkx_edges(G, pos, [(u, v)], 
                                      width=width, alpha=alpha,
                                      edge_color=edge_color,
                                      arrows=True, arrowsize=arrow_size,
                                      arrowstyle='-|>',
                                      connectionstyle=f'arc3,rad={rad}',
                                      node_size=4500, ax=ax)
                
                rev_idx = list(G.edges()).index((v, u))
                width_rev = widths[rev_idx]
                alpha_rev = alphas[rev_idx]
                nx.draw_networkx_edges(G, pos, [(v, u)],
                                      width=width_rev, alpha=alpha_rev,
                                      edge_color=edge_color,
                                      arrows=True, arrowsize=arrow_size,
                                      arrowstyle='-|>',
                                      connectionstyle=f'arc3,rad={rad}',
                                      node_size=4500, ax=ax)
            elif not has_reverse or u > v:
                if not has_reverse:
                    nx.draw_networkx_edges(G, pos, [(u, v)],
                                          width=width, alpha=alpha,
                                          edge_color=edge_color,
                                          arrows=True, arrowsize=arrow_size,
                                          arrowstyle='-|>',
                                          connectionstyle='arc3,rad=0.0',
                                          node_size=4500, ax=ax)
    
    # Legend (Larger font, compact placement)
    legend_elements = [
        mpatches.Patch(facecolor='#E64B35', label='CEN', edgecolor='white'),
        mpatches.Patch(facecolor='#4DBBD5', label='Salience', edgecolor='white'),
        mpatches.Patch(facecolor='#00A087', label='DMN', edgecolor='white')
    ]
    
    # Place legend specifically to not overlap with circular nodes (e.g., top right corner)
    # Using bbox_to_anchor to control placement more precisely
    ax.legend(handles=legend_elements, loc='upper right', 
              bbox_to_anchor=(1.0, 1.0),
              fontsize=16, frameon=True, fancybox=True,
              framealpha=0.9) # 150% of typical 10-11pt
    
    if title:
        ax.set_title(title, fontsize=22, fontweight='bold', pad=10)
    
    ax.axis('off')
    # Tighter limits to reduce white space
    ax.set_xlim(-1.1, 1.1) 
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()

def main():
    # 1. GCM Figure
    gcm_ts = "11272025173313"
    print(f"Generating Compact GCM Figure for {gcm_ts}...")
    try:
        gcm_rates = load_gcm_results(gcm_ts)
        plot_network_compact(
            gcm_rates, 
            threshold=0.3, 
            output_path=f"gcm_roebroeck/{gcm_ts}/enhanced_plots/network_circular_thresh_30_compact.png",
            title="GCM (thresh > 30%)"
        )
    except Exception as e:
        print(f"Error generating GCM figure: {e}")

    # 2. RASL Figure
    rasl_ts = "11262025164900"
    print(f"Generating Compact RASL Figure for {rasl_ts}...")
    try:
        rasl_rates = load_fmri_results(rasl_ts)
        plot_network_compact(
            rasl_rates, 
            threshold=0.5, 
            output_path=f"fbirn_results/{rasl_ts}/combined/enhanced_plots/network_thresh_50_compact.png",
            title="RnR (thresh > 50%)"
        )
    except Exception as e:
        print(f"Error generating RASL figure: {e}")

if __name__ == "__main__":
    main()

