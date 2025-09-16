import datadocket as dd
import matplotlib.pyplot as plt
import numpy as np

def relative_directions_plot(
    average_embeddings_file: str,
    pca_model_file: str,
    variations: list,
    show: bool = False) -> bool:
    """
    """
    
    variation_labels = [variation.replace("_change", "").capitalize() for variation in variations]
    variation_labels.append("Original Question")

    color = "grey"
    shapes = ["o", "s", "d", "v", "p"]
    marker_size = 20

    # load statistics
    directions_stats = dd.load.Json("data/relative_directions_stats.json")
    distances_stats = dd.load.Json("data/distances_stats.json")

    # Create the arrow plot
    fig, ax = plt.subplots(figsize=(12, 12))

    # Center point (origin)
    center_x, center_y = 0, 0

    # Plot the center point
    ax.plot(center_x, center_y, 'ko', markersize=marker_size, label='Original Question')

    # Arrow data
    arrows_data = []

    for i, variation in enumerate(variations):
        # Get angle in degrees and convert to radians or plotting
        angle_deg = directions_stats[variation]["average"]
        angle_rad = np.radians(angle_deg)
        
        # Get distance (length of arrow)
        distance = distances_stats[variation]["average"]
        
        # Calculate arrow end point
        end_x = center_x + distance * np.cos(angle_rad)
        end_y = center_y + distance * np.sin(angle_rad)
        
        # Store arrow data for legend
        arrows_data.append({
            'variation': variation,
            'label': variation_labels[i],
            'color': color,
            'angle': angle_deg,
            'distance': distance
        })
        
        # Draw the marker
        ax.plot(end_x, end_y, marker=shapes[i+1], markersize=marker_size, color=color, 
                alpha=0.8, markeredgecolor='black', markeredgewidth=1)

    # Set equal aspect ratio and limits
    max_distance = max([data['distance'] for data in arrows_data])
    # Very minimal padding around the markers
    padding = max_distance * -0.05
    ax.set_xlim(-max_distance - padding, max_distance + padding)
    ax.set_ylim(-max_distance - padding, max_distance + padding)
    ax.set_aspect('equal')

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=0.5)
    ax.axvline(x=0, color='k', linewidth=0.5)

    # Add angle reference lines for all quadrants
    for angle in range(0, 360, 30):  # Every 30 degrees from 0 to 330
        angle_rad = np.radians(angle)
        x_end = max_distance * 0.8 * np.cos(angle_rad)
        y_end = max_distance * 0.8 * np.sin(angle_rad)
        ax.plot([0, x_end], [0, y_end], 'k--', alpha=0.2, linewidth=0.5)
        ax.text(x_end * 1.1, y_end * 1.1, f'{angle}°', fontsize=8, alpha=0.6)

    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker=shape, color=color, linestyle='None', 
                                markersize=10, markeredgecolor='black', markeredgewidth=1, 
                                label=label) for shape, label in zip(shapes[1:], variation_labels)]
    legend_elements.insert(0, plt.Line2D([0], [0], marker='o', color='k', linestyle='None', 
                                        markersize=10, label='Original Question'))

    ax.legend(handles=legend_elements, title="Variation Types", 
            bbox_to_anchor=(1.05, 1), loc='upper left')

    # Labels and title
    ax.set_xlabel('Distance (Embedding Space)', fontsize=12)
    ax.set_ylabel('Distance (Embedding Space)', fontsize=12)
    ax.set_title('Relative Embedding Position to the original question\n(Marker Position = Average Relative Distance & Angle)', 
                fontsize=14, pad=30)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.savefig("plots/relative_directions.png", bbox_inches='tight', dpi=300)
    if show:
        plt.show()

    return True