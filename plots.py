import datadocket as dd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import pickle


def absolute_directions_plot(
    variations: list,
    show: bool = False) -> bool:
    """
    """
    variations = variations[1:]
    
    variation_labels = [variation.replace("_change", "").capitalize() for variation in variations]

    color = "grey"
    shapes = ["o", "s", "d", "v", "p"]
    marker_size = 20

    # load statistics
    directions_stats = dd.load.Json("data/directions_stats.json")
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


def relative_directions_plot(
    variations: list,
    show: bool = False) -> bool:
    """
    """
    variations = variations[1:]
    
    variation_labels = [variation.replace("_change", "").capitalize() for variation in variations]

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

def box_plot(
    stat_file: str,
    variations: list,
    colors: list = ['lightcoral', 'lightgreen', 'gold', 'plum'],
    show: bool = False) -> bool:
    """
    """
    
    # load distances stats
    distances_stats = dd.load.Json(stat_file)

    # Create boxplot visualization
    plt.figure(figsize=(12, 8))

    # Get data for variations (excluding "question") and sort by average
    variation_data = []
    for var in variations[1:]:
        stats = distances_stats[var]
        variation_data.append({
            'name': var,
            'label': var.replace("_", "\n").title(),
            'stats': stats,
            'average': stats["average"]
        })

    # Sort by average value (lowest to highest)
    variation_data.sort(key=lambda x: x['average'])

    # Extract sorted data
    variation_labels = [item['label'] for item in variation_data]
    boxplot_data = []
    averages = []

    for i, item in enumerate(variation_data):
        stats = item['stats']
        # Create a pseudo-boxplot using the statistics
        # We'll represent min, Q1 (approximated), median, Q3 (approximated), max
        q1_approx = stats["min"] + (stats["median"] - stats["min"]) * 0.25
        q3_approx = stats["median"] + (stats["max"] - stats["median"]) * 0.25
        
        boxplot_data.append([
            stats["min"],      # Lower whisker
            q1_approx,         # Q1
            stats["median"],   # Median
            q3_approx,         # Q3
            stats["max"]       # Upper whisker
        ])
        averages.append(stats["average"])

    # Create boxplot
    bp = plt.boxplot(boxplot_data, labels=variation_labels, patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5))

    # Color the boxes differently
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    # Add average points (no line connection)
    x_positions = range(1, len(variation_labels) + 1)
    plt.scatter(x_positions, averages, color='darkblue', s=100, 
            marker='o', label='Average', zorder=5)

    plt.ylabel("Distance", fontsize=12)
    plt.title("Distance Distribution by Variation Type (Sorted by Average)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Create legend for colors (variation types) - in the new order
    color_legend_elements = []
    for i, item in enumerate(variation_data):
        color_legend_elements.append(
            plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=item['label'])
        )

    # Add the legend
    plt.legend(handles=color_legend_elements, title="Variation Types", 
            bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f"plots/{stat_file.split('/')[-1].split('.')[0]}_boxplot.png", bbox_inches='tight', dpi=300)
    if show:
        plt.show()

    return True


def pca_plot_all(
    average_embeddings_file: str,
    pca_model_file: str,
    variations: list,
    shapes: list = ["o", "s", "d", "v", "p"],
    show: bool = False) -> bool:
    """
    """
    # load average embeddings
    average_embeddings = dd.load.Json(average_embeddings_file)

    # load PCA model
    with open(pca_model_file, "rb") as f:
        pca = pickle.load(f)

    # Create a continuous color map for different questions (works well for many questions)
    num_questions = len(average_embeddings)
    cmap = cm.get_cmap('nipy_spectral')
    colors = [cmap(i / (num_questions - 1)) for i in range(num_questions)]

    plt.figure(figsize=(15, 10))

    # Plot each question with different colors
    for question_idx, question_data in enumerate(average_embeddings):
        # Extract only the embedding vectors (not the 'id' field)
        embeddings = [question_data[variation] for variation in variations]
        
        # Convert to numpy array
        embeddings = np.array(embeddings)
        
        # Transform to PCA space
        embeddings_pca = pca.transform(embeddings)
            
        # Plot each variation with different shapes but same color for the question
        for variation_idx, (variation_embedding_pca, shape) in enumerate(zip(embeddings_pca, shapes)):
            plt.scatter(variation_embedding_pca[0], variation_embedding_pca[1], 
                    s=100, alpha=0.7, marker=shape, 
                    color=colors[question_idx], 
                    label=f"Q{question_data['id']}" if variation_idx == 0 else "")

    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA of Question Variations (Each Question = Different Color)')

    # Create shape legend elements
    shape_legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Question', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='s', color='w', label='Synonym Change', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='d', color='w', label='Antonym Change', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='v', color='w', label='Paraphrase Change', markerfacecolor='gray', markersize=10),
        Line2D([0], [0], marker='p', color='w', label='Letter Change', markerfacecolor='gray', markersize=10),
    ]

    # create legend
    plt.legend(handles=shape_legend_elements, title="Variation Types", 
                        loc='upper left', ncol=1, fontsize=9, framealpha=0.8)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"plots/{average_embeddings_file.split('/')[-1].split('.')[0]}_pca_plot.png", bbox_inches='tight')
    if show:
        plt.show()
    return True


