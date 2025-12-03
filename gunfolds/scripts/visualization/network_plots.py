"""
Unified network plotting module for GCM, RASL, and fMRI results.

This module consolidates plotting functionality from:
- plot_gcm_enhanced.py
- plot_fmri_enhanced.py  
- plot_manuscript_compact.py

Usage:
    # For GCM results
    python network_plots.py --source gcm --timestamp 11272025173313
    
    # For fMRI/RASL results
    python network_plots.py --source fmri --timestamp 11262025164900 --groups combined 0 1
    
    # For manuscript compact figures
    python network_plots.py --source gcm --timestamp 11272025173313 --compact
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_gcm_results(timestamp):
    """Load GCM results from timestamp directory"""
    base_dir = f"gcm_roebroeck/{timestamp}"
    
    rate_file = f"{base_dir}/csv/group_edge_rate.csv"
    hits_file = f"{base_dir}/csv/group_edge_hits.csv"
    
    if not os.path.exists(rate_file):
        raise FileNotFoundError(f"Results not found: {rate_file}")
    
    edge_rate = pd.read_csv(rate_file, index_col=0)
    edge_hits = pd.read_csv(hits_file, index_col=0)
    
    return edge_rate, edge_hits


def load_fmri_results(timestamp, group='combined'):
    """Load fMRI RASL results from timestamp directory"""
    base_dir = f"fbirn_results/{timestamp}"
    
    if group == 'combined':
        csv_file = f"{base_dir}/combined/group_edge_counts_combined.csv"
        summary_file = f"{base_dir}/combined/analysis_summary.csv"
    else:
        csv_file = f"{base_dir}/group_{group}/group_edge_counts_{group}.csv"
        summary_file = f"{base_dir}/combined/analysis_summary.csv"
    
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Results not found: {csv_file}")
    
    # Load edge counts (long format: src, dst, count)
    edge_df = pd.read_csv(csv_file)
    
    # Get unique nodes
    nodes = sorted(list(set(edge_df['src'].unique()) | set(edge_df['dst'].unique())))
    n_nodes = len(nodes)
    
    # Convert to matrix format
    edge_matrix = np.zeros((n_nodes, n_nodes))
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    
    for _, row in edge_df.iterrows():
        i = node_to_idx[row['src']]
        j = node_to_idx[row['dst']]
        edge_matrix[i, j] = row['count']
    
    # Load summary to get number of subjects/solutions
    if os.path.exists(summary_file):
        summary_df = pd.read_csv(summary_file)
        if group == 'combined':
            n_solutions = summary_df['Num_Solutions'].sum()
        else:
            n_solutions = summary_df[summary_df['Group'] == int(group)]['Num_Solutions'].sum()
    else:
        # Estimate from max count
        n_solutions = edge_matrix.max()
    
    # Convert to frequency (rate)
    edge_rate = edge_matrix / n_solutions
    
    # Create DataFrame
    edge_rate_df = pd.DataFrame(edge_rate, index=nodes, columns=nodes)
    edge_count_df = pd.DataFrame(edge_matrix, index=nodes, columns=nodes)
    
    return edge_rate_df, edge_count_df, n_solutions


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_network_circular(edge_rate, threshold=0.3, output_path=None, title=None, 
                         n_solutions=None, compact=False):
    """
    Create circular network plot with threshold.
    
    Args:
        edge_rate: DataFrame with edge frequencies
        threshold: Minimum edge frequency to display
        output_path: Path to save figure
        title: Figure title
        n_solutions: Total number of solutions (for info)
        compact: Use compact style for manuscripts
    """
    # Standardize node order for consistency
    standard_order = ['rPPC', 'rDLPFC', 'rFIC', 'ACC', 'PCC', 'VMPFC']
    available_nodes = [n for n in standard_order if n in edge_rate.columns]
    extra_nodes = [n for n in edge_rate.columns if n not in standard_order]
    final_order = available_nodes + extra_nodes
    
    if available_nodes:
        edge_rate = edge_rate.reindex(index=final_order, columns=final_order)
    
    node_names = list(edge_rate.columns)
    N = len(node_names)
    
    # Network colors by functional system
    network_colors = {
        'rPPC': '#E64B35',    # CEN - Red
        'rDLPFC': '#E64B35',  # CEN - Red
        'rFIC': '#4DBBD5',    # Salience - Blue
        'ACC': '#4DBBD5',     # Salience - Blue
        'PCC': '#00A087',     # DMN - Green
        'VMPFC': '#00A087'    # DMN - Green
    }
    
    # Create directed graph
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(i, label=name, color=network_colors.get(name, '#666666'))
    
    # Add edges above threshold
    edge_matrix = edge_rate.to_numpy()
    for i in range(N):
        for j in range(N):
            if i != j and edge_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=edge_matrix[i, j])
    
    # Check if we have edges
    if G.number_of_edges() == 0:
        print(f"Warning: No edges above threshold {threshold} for {title}")
        return
    
    # Layout
    if compact:
        figsize = (8, 8)
        node_size = 4500
        font_size = 18
        arrow_size = 35
        radius = 0.85
        widths_base = 1.5
    else:
        figsize = (10, 10)
        node_size = 3000
        font_size = 14
        arrow_size = 25
        radius = 1.0
        widths_base = 1.0
    
    # Fixed circular layout - start at top, go clockwise
    angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, N, endpoint=False)
    pos = {i: (np.cos(angles[i])*radius, np.sin(angles[i])*radius) for i in range(N)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')
    
    # Node colors
    node_colors = [G.nodes[i]['color'] for i in range(N)]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors,
                           edgecolors='white', linewidths=3, ax=ax)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, 
                           labels={i: node_names[i] for i in range(N)},
                           font_size=font_size, font_weight='bold', ax=ax)
    
    # Edge properties
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    
    if len(weights) > 0:
        # Normalize weights for width and alpha
        w_min, w_max = weights.min(), weights.max()
        if w_max > w_min:
            norm_weights = (weights - w_min) / (w_max - w_min)
        else:
            norm_weights = np.ones(len(weights))
        
        widths = widths_base + norm_weights * 8
        alphas = 0.4 + norm_weights * 0.5 if not compact else 0.5 + norm_weights * 0.5
        
        # Draw edges
        for (u, v), width, alpha in zip(G.edges(), widths, alphas):
            # Check if bidirectional
            has_reverse = G.has_edge(v, u)
            
            if has_reverse and u < v:
                # Draw curved bidirectional edges
                rad = 0.3
                edge_color = '#333333'
                
                # Forward edge
                nx.draw_networkx_edges(G, pos, [(u, v)], 
                                      width=width, alpha=alpha,
                                      edge_color=edge_color,
                                      arrows=True, arrowsize=arrow_size,
                                      arrowstyle='-|>',
                                      connectionstyle=f'arc3,rad={rad}',
                                      node_size=node_size, ax=ax)
                
                # Reverse edge
                rev_idx = list(G.edges()).index((v, u))
                width_rev = widths[rev_idx]
                alpha_rev = alphas[rev_idx]
                nx.draw_networkx_edges(G, pos, [(v, u)],
                                      width=width_rev, alpha=alpha_rev,
                                      edge_color=edge_color,
                                      arrows=True, arrowsize=arrow_size,
                                      arrowstyle='-|>',
                                      connectionstyle=f'arc3,rad={rad}',
                                      node_size=node_size, ax=ax)
            elif not has_reverse or u > v:
                # Unidirectional edge (or already drawn)
                if not has_reverse:
                    nx.draw_networkx_edges(G, pos, [(u, v)],
                                          width=width, alpha=alpha,
                                          edge_color='#333333',
                                          arrows=True, arrowsize=arrow_size,
                                          arrowstyle='-|>',
                                          connectionstyle='arc3,rad=0.0',
                                          node_size=node_size, ax=ax)
    
    # Legend
    if compact:
        legend_elements = [
            mpatches.Patch(facecolor='#E64B35', label='CEN', edgecolor='white'),
            mpatches.Patch(facecolor='#4DBBD5', label='Salience', edgecolor='white'),
            mpatches.Patch(facecolor='#00A087', label='DMN', edgecolor='white')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(1.0, 1.0),
                  fontsize=16, frameon=True, fancybox=True, framealpha=0.9)
    else:
        legend_elements = [
            mpatches.Patch(facecolor='#E64B35', label='CEN (Central Executive)', edgecolor='white'),
            mpatches.Patch(facecolor='#4DBBD5', label='Salience Network', edgecolor='white'),
            mpatches.Patch(facecolor='#00A087', label='DMN (Default Mode)', edgecolor='white')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11, frameon=True, fancybox=True)
    
    # Title
    if title:
        title_size = 22 if compact else 16
        title_pad = 10 if compact else 20
        ax.set_title(title, fontsize=title_size, fontweight='bold', pad=title_pad)
    else:
        ax.set_title(f'Network (threshold > {threshold:.1%})', fontsize=16, fontweight='bold', pad=20)
    
    ax.axis('off')
    limit = 1.1 if compact else 1.4
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    plt.tight_layout()
    
    if output_path:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_network_spring(edge_rate, threshold=0.3, output_path=None):
    """Create spring layout network plot"""
    node_names = list(edge_rate.columns)
    N = len(node_names)
    
    # Network colors
    network_colors = {
        'rPPC': '#E64B35', 'rDLPFC': '#E64B35',
        'rFIC': '#4DBBD5', 'ACC': '#4DBBD5',
        'PCC': '#00A087', 'VMPFC': '#00A087'
    }
    
    G = nx.DiGraph()
    for i, name in enumerate(node_names):
        G.add_node(i, label=name, color=network_colors.get(name, '#666666'))
    
    edge_matrix = edge_rate.to_numpy()
    for i in range(N):
        for j in range(N):
            if i != j and edge_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=edge_matrix[i, j])
    
    if G.number_of_edges() == 0:
        return
    
    # Spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    fig, ax = plt.subplots(figsize=(12, 10), facecolor='white')
    
    node_colors = [G.nodes[i]['color'] for i in range(N)]
    
    # Draw
    nx.draw_networkx_nodes(G, pos, node_size=4000, node_color=node_colors,
                           edgecolors='white', linewidths=3, ax=ax)
    nx.draw_networkx_labels(G, pos,
                           labels={i: node_names[i] for i in range(N)},
                           font_size=14, font_weight='bold', ax=ax)
    
    weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
    if len(weights) > 0:
        norm_weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
        widths = 1 + norm_weights * 8
        alphas = 0.4 + norm_weights * 0.5
        
        for (u, v), width, alpha in zip(G.edges(), widths, alphas):
            nx.draw_networkx_edges(G, pos, [(u, v)],
                                  width=width, alpha=alpha,
                                  edge_color='#333333',
                                  arrows=True, arrowsize=30,
                                  arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1',
                                  node_size=4000, ax=ax)
    
    legend_elements = [
        mpatches.Patch(facecolor='#E64B35', label='CEN', edgecolor='white'),
        mpatches.Patch(facecolor='#4DBBD5', label='Salience', edgecolor='white'),
        mpatches.Patch(facecolor='#00A087', label='DMN', edgecolor='white')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11)
    
    ax.set_title(f'Network - Spring Layout (threshold > {threshold:.1%})',
                fontsize=16, fontweight='bold', pad=20)
    ax.axis('off')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_heatmap_enhanced(edge_rate, output_path=None, title=None):
    """Create enhanced heatmap with better colors and annotations"""
    fig, ax = plt.subplots(figsize=(10, 9), facecolor='white')
    
    # Mask diagonal
    matrix = edge_rate.to_numpy()
    mask = np.eye(len(matrix), dtype=bool)
    matrix_masked = np.ma.masked_where(mask, matrix)
    
    # Plot with better colormap
    im = ax.imshow(matrix_masked, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Edge Frequency', rotation=270, labelpad=20, fontsize=12)
    
    # Ticks and labels
    node_names = list(edge_rate.columns)
    ax.set_xticks(range(len(node_names)))
    ax.set_yticks(range(len(node_names)))
    ax.set_xticklabels(node_names, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(node_names, fontsize=11)
    
    ax.set_xlabel('Target', fontsize=13, fontweight='bold')
    ax.set_ylabel('Source', fontsize=13, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title('Edge Frequencies - Complete Matrix', fontsize=14, fontweight='bold', pad=15)
    
    # Add text annotations for high values
    annotation_threshold = 0.4 if matrix.max() < 0.8 else 0.7
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i != j:
                val = matrix[i, j]
                if val > annotation_threshold:
                    color = 'white' if val > (annotation_threshold + 0.05) else 'black'
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=9, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(len(node_names)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(node_names)) - 0.5, minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=2)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_top_connections(edge_rate, n_top=15, output_path=None, title=None):
    """Create bar plot of top N connections"""
    node_names = list(edge_rate.columns)
    matrix = edge_rate.to_numpy()
    
    # Get all connections
    connections = []
    for i in range(len(node_names)):
        for j in range(len(node_names)):
            if i != j:
                connections.append((node_names[i], node_names[j], matrix[i, j]))
    
    # Sort and get top N
    connections.sort(key=lambda x: x[2], reverse=True)
    top_connections = connections[:n_top]
    
    # Create labels and values
    labels = [f"{src} → {tgt}" for src, tgt, _ in top_connections]
    values = [val for _, _, val in top_connections]
    
    # Colors based on network pairs
    colors = []
    for src, tgt, _ in top_connections:
        if 'FIC' in src or 'ACC' in src:
            if 'DLPFC' in tgt or 'PPC' in tgt:
                colors.append('#9467BD')  # Salience → CEN (purple)
            elif 'PCC' in tgt or 'VMPFC' in tgt:
                colors.append('#FF7F0E')  # Salience → DMN (orange)
            else:
                colors.append('#4DBBD5')  # Within Salience (blue)
        elif 'DLPFC' in src or 'PPC' in src:
            if 'PCC' in tgt or 'VMPFC' in tgt:
                colors.append('#8C564B')  # CEN → DMN (brown)
            else:
                colors.append('#E64B35')  # Within/from CEN (red)
        elif 'PCC' in src or 'VMPFC' in src:
            colors.append('#00A087')  # DMN connections (green)
        else:
            colors.append('#666666')
    
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    bars = ax.barh(range(len(labels)), values, color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Edge Frequency', fontsize=12, fontweight='bold')
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    else:
        ax.set_title(f'Top {n_top} Strongest Connections', fontsize=14, fontweight='bold', pad=15)
    
    ax.set_xlim(0, 1)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        label_format = f'{val:.1%}' if val >= 0.01 else f'{val:.2%}'
        ax.text(val + 0.01, i, label_format, va='center', fontsize=9)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


def plot_group_comparison(edge_rate_0, edge_rate_1, threshold=0.5, output_path=None):
    """Create side-by-side comparison of group 0 vs group 1"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), facecolor='white')
    
    node_names = list(edge_rate_0.columns)
    
    for ax, edge_rate, group_name in [(ax1, edge_rate_0, 'Group 0 (Control)'), 
                                       (ax2, edge_rate_1, 'Group 1 (Patient)')]:
        N = len(node_names)
        
        network_colors = {
            'rPPC': '#E64B35', 'rDLPFC': '#E64B35',
            'rFIC': '#4DBBD5', 'ACC': '#4DBBD5',
            'PCC': '#00A087', 'VMPFC': '#00A087'
        }
        
        G = nx.DiGraph()
        for i, name in enumerate(node_names):
            G.add_node(i, label=name, color=network_colors.get(name, '#666666'))
        
        edge_matrix = edge_rate.to_numpy()
        for i in range(N):
            for j in range(N):
                if i != j and edge_matrix[i, j] > threshold:
                    G.add_edge(i, j, weight=edge_matrix[i, j])
        
        if G.number_of_edges() == 0:
            ax.text(0.5, 0.5, f'No edges > {threshold:.1%}', 
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            ax.axis('off')
            ax.set_title(group_name, fontsize=14, fontweight='bold')
            continue
        
        angles = np.linspace(np.pi/2, np.pi/2 - 2*np.pi, N, endpoint=False)
        pos = {i: (np.cos(angles[i]), np.sin(angles[i])) for i in range(N)}
        
        node_colors = [G.nodes[i]['color'] for i in range(N)]
        
        nx.draw_networkx_nodes(G, pos, node_size=2500, node_color=node_colors,
                               edgecolors='white', linewidths=3, ax=ax)
        nx.draw_networkx_labels(G, pos,
                               labels={i: node_names[i] for i in range(N)},
                               font_size=12, font_weight='bold', ax=ax)
        
        weights = np.array([G[u][v]['weight'] for u, v in G.edges()])
        if len(weights) > 0:
            w_min, w_max = weights.min(), weights.max()
            if w_max > w_min:
                norm_weights = (weights - w_min) / (w_max - w_min)
            else:
                norm_weights = np.ones(len(weights))
            
            widths = 1 + norm_weights * 8
            alphas = 0.4 + norm_weights * 0.5
            
            for (u, v), width, alpha in zip(G.edges(), widths, alphas):
                has_reverse = G.has_edge(v, u)
                
                if has_reverse and u < v:
                    rad = 0.3
                    nx.draw_networkx_edges(G, pos, [(u, v)],
                                          width=width, alpha=alpha,
                                          edge_color='#333333',
                                          arrows=True, arrowsize=20,
                                          arrowstyle='-|>',
                                          connectionstyle=f'arc3,rad={rad}',
                                          node_size=2500, ax=ax)
                    
                    rev_idx = list(G.edges()).index((v, u))
                    width_rev = widths[rev_idx]
                    alpha_rev = alphas[rev_idx]
                    nx.draw_networkx_edges(G, pos, [(v, u)],
                                          width=width_rev, alpha=alpha_rev,
                                          edge_color='#333333',
                                          arrows=True, arrowsize=20,
                                          arrowstyle='-|>',
                                          connectionstyle=f'arc3,rad={rad}',
                                          node_size=2500, ax=ax)
                elif not has_reverse:
                    nx.draw_networkx_edges(G, pos, [(u, v)],
                                          width=width, alpha=alpha,
                                          edge_color='#333333',
                                          arrows=True, arrowsize=20,
                                          arrowstyle='-|>',
                                          connectionstyle='arc3,rad=0.0',
                                          node_size=2500, ax=ax)
        
        ax.axis('off')
        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_title(group_name, fontsize=14, fontweight='bold')
    
    # Overall title
    fig.suptitle(f'Networks by Group (threshold > {threshold:.1%})', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Unified network plotting for GCM and fMRI results')
    parser.add_argument('--source', required=True, choices=['gcm', 'fmri', 'rasl'],
                       help='Data source type')
    parser.add_argument('-t', '--timestamp', required=True, help='Timestamp directory')
    parser.add_argument('--thresholds', nargs='+', type=float, default=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       help='Thresholds for network plots')
    parser.add_argument('--groups', nargs='+', default=['combined'],
                       help='Groups to plot for fMRI (combined, 0, 1)')
    parser.add_argument('--compact', action='store_true',
                       help='Use compact manuscript style')
    parser.add_argument('--spring-layout', action='store_true',
                       help='Also generate spring layout plots (GCM only)')
    args = parser.parse_args()
    
    # Normalize source naming
    if args.source == 'rasl':
        args.source = 'fmri'
    
    # Process based on source
    if args.source == 'gcm':
        print(f"Loading GCM results from: gcm_roebroeck/{args.timestamp}")
        edge_rate, edge_hits = load_gcm_results(args.timestamp)
        
        output_dir = f"gcm_roebroeck/{args.timestamp}/enhanced_plots"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nCreating visualizations...")
        print(f"Output directory: {output_dir}")
        
        # 1. Network plots at different thresholds
        for threshold in args.thresholds:
            n_edges = np.sum(edge_rate.to_numpy() > threshold)
            if n_edges > 0:
                print(f"  - Circular network plot (threshold={threshold:.1%}, {n_edges} edges)")
                title = f'GCM Network (threshold > {threshold:.1%})'
                plot_network_circular(edge_rate, threshold=threshold,
                                    output_path=f"{output_dir}/network_circular_thresh_{int(threshold*100)}.png",
                                    title=title, compact=args.compact)
                
                if args.spring_layout:
                    print(f"  - Spring layout plot (threshold={threshold:.1%})")
                    plot_network_spring(edge_rate, threshold=threshold,
                                      output_path=f"{output_dir}/network_spring_thresh_{int(threshold*100)}.png")
            else:
                print(f"  - Skipping threshold={threshold:.1%} (no edges)")
        
        # 2. Enhanced heatmap
        print(f"  - Enhanced heatmap")
        plot_heatmap_enhanced(edge_rate, 
                            output_path=f"{output_dir}/heatmap_enhanced.png",
                            title='GCM Edge Frequencies - Complete Matrix')
        
        # 3. Top connections
        print(f"  - Top connections bar plot")
        plot_top_connections(edge_rate, n_top=15,
                           output_path=f"{output_dir}/top_connections.png",
                           title='Top 15 Strongest GCM Connections')
        
        # 4. Summary statistics
        print(f"\n{'='*60}")
        print("SUMMARY STATISTICS")
        print(f"{'='*60}")
        
        matrix = edge_rate.to_numpy()
        np.fill_diagonal(matrix, 0)
        
        print(f"Total subjects: {edge_hits.max().max():.0f}")
        print(f"Mean edge frequency: {matrix[matrix > 0].mean():.2%}")
        print(f"Max edge frequency: {matrix.max():.2%}")
        for thresh in [0.3, 0.4, 0.5]:
            print(f"Edges > {thresh:.0%}: {np.sum(matrix > thresh)}")
    
    elif args.source == 'fmri':
        print(f"Loading fMRI results from: fbirn_results/{args.timestamp}")
        
        output_base = f"fbirn_results/{args.timestamp}"
        
        # Process each group
        for group in args.groups:
            print(f"\n{'='*60}")
            print(f"Processing: {group.upper()}")
            print(f"{'='*60}")
            
            try:
                edge_rate, edge_count, n_solutions = load_fmri_results(args.timestamp, group)
                
                output_dir = os.path.join(output_base, f"{group}/enhanced_plots" if group != 'combined' 
                                         else "combined/enhanced_plots")
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"Output directory: {output_dir}")
                print(f"Total solutions: {n_solutions:.0f}")
                
                # Statistics
                matrix = edge_rate.to_numpy()
                np.fill_diagonal(matrix, 0)
                print(f"Mean edge frequency: {matrix[matrix > 0].mean():.2%}")
                print(f"Max edge frequency: {matrix.max():.2%}")
                
                # 1. Network plots at different thresholds
                for threshold in args.thresholds:
                    n_edges = np.sum(matrix > threshold)
                    if n_edges > 0:
                        print(f"  - Network plot (threshold={threshold:.1%}, {n_edges} edges)")
                        title = f'RASL Network - {group.upper()} (threshold > {threshold:.1%})'
                        plot_network_circular(edge_rate, threshold=threshold,
                                            output_path=f"{output_dir}/network_thresh_{int(threshold*100)}.png",
                                            title=title, n_solutions=n_solutions, compact=args.compact)
                    else:
                        print(f"  - Skipping threshold={threshold:.1%} (no edges)")
                
                # 2. Enhanced heatmap
                print(f"  - Enhanced heatmap")
                title = f'RASL Edge Frequencies - {group.upper()}'
                plot_heatmap_enhanced(edge_rate, 
                                    output_path=f"{output_dir}/heatmap_enhanced.png",
                                    title=title)
                
                # 3. Top connections
                print(f"  - Top connections bar plot")
                title = f'Top 15 Strongest RASL Connections - {group.upper()}'
                plot_top_connections(edge_rate, n_top=15,
                                   output_path=f"{output_dir}/top_connections.png",
                                   title=title)
                
            except Exception as e:
                print(f"Error processing {group}: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. Group comparison plots
        if '0' in args.groups and '1' in args.groups:
            print(f"\n{'='*60}")
            print("Creating Group Comparisons")
            print(f"{'='*60}")
            
            edge_rate_0, _, _ = load_fmri_results(args.timestamp, '0')
            edge_rate_1, _, _ = load_fmri_results(args.timestamp, '1')
            
            comparison_dir = os.path.join(output_base, "group_comparison")
            os.makedirs(comparison_dir, exist_ok=True)
            
            for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
                print(f"  - Comparison plot (threshold={threshold:.1%})")
                plot_group_comparison(edge_rate_0, edge_rate_1, threshold=threshold,
                                    output_path=f"{comparison_dir}/comparison_thresh_{int(threshold*100)}.png")
    
    print(f"\n{'='*60}")
    print(f"✓ All plots saved!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

