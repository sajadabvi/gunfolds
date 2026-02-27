"""
Combine multiple plots into a single SVG figure for manuscript
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
import os

# Paths to the images
img_paths = {
    'A': 'gunfolds/scripts/gcm_roebroeck/11272025173313/enhanced_plots/network_circular_thresh_30.png',
    'B': 'gunfolds/scripts/gcm_roebroeck/11272025173313/enhanced_plots/top_connections.png',
    'C': 'gunfolds/scripts/fbirn_results/11262025164900/combined/enhanced_plots/network_thresh_50.png',
    'D': 'gunfolds/scripts/fbirn_results/11262025164900/combined/enhanced_plots/top_connections.png'
}

def create_combined_figure(output_path='combined_figure.svg'):
    # Check if files exist
    for key, path in img_paths.items():
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            # Try fixing path if running from root
            alt_path = path.replace('gunfolds/scripts/', '')
            if os.path.exists(alt_path):
                img_paths[key] = alt_path
                print(f"Found at: {alt_path}")
            else:
                # Try absolute path based on previous context
                abs_prefix = '/Users/mabavisani/code_local/mygit/gunfolds/'
                abs_path = os.path.join(abs_prefix, path)
                if os.path.exists(abs_path):
                    img_paths[key] = abs_path
                    print(f"Found at: {abs_path}")
                else:
                    return

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(20, 16), facecolor='white')
    gs = GridSpec(2, 2, figure=fig, wspace=0.1, hspace=0.15)
    
    axes = {
        'A': fig.add_subplot(gs[0, 0]),
        'B': fig.add_subplot(gs[0, 1]),
        'C': fig.add_subplot(gs[1, 0]),
        'D': fig.add_subplot(gs[1, 1])
    }
    
    titles = {
        'A': 'GCM Network (Temporal Causality)',
        'B': 'GCM Top Connections',
        'C': 'RASL Network (Structural Causality)',
        'D': 'RASL Top Connections'
    }
    
    # Load and display images
    for key, ax in axes.items():
        img = mpimg.imread(img_paths[key])
        ax.imshow(img)
        ax.axis('off')
        
        # Add panel labels (A, B, C, D)
        ax.text(-0.05, 1.05, key, transform=ax.transAxes, 
                fontsize=24, fontweight='bold', va='top', ha='right')
        
        # Add titles (optional, images already have titles)
        # ax.set_title(titles[key], fontsize=14, pad=10)
    
    # Save
    plt.savefig(output_path, format='svg', bbox_inches='tight', dpi=300)
    print(f"Saved combined figure to: {output_path}")
    
    # Also save as PNG for preview
    png_path = output_path.replace('.svg', '.png')
    plt.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
    print(f"Saved preview to: {png_path}")

if __name__ == "__main__":
    create_combined_figure('manuscript_combined_figure.svg')

