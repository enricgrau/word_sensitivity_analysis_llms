import math

import datadocket as dd
import matplotlib.pyplot as plt
import numpy as np


def calculate_angle_x(point_a, point_b):
  """
  Calculates the angle of the line segment from point_a to point_b
  with respect to the positive horizontal axis.

  Args:
    point_a: A tuple or list representing the coordinates of point A (x_a, y_a).
    point_b: A tuple or list representing the coordinates of point B (x_b, y_b).

  Returns:
    The angle in degrees, ranging from -180 to 180.
  """
  # Unpack the coordinates from the input points
  x_a, y_a = point_a
  x_b, y_b = point_b

  # Calculate the difference in the x and y coordinates (the vector components)
  delta_x = x_b - x_a
  delta_y = y_b - y_a

  # Use atan2 to get the angle in radians.
  # math.atan2(y, x) is preferred over math.atan(y/x) because it
  # correctly handles all four quadrants and avoids division by zero.
  angle_radians = math.atan2(delta_y, delta_x)

  # Convert the angle from radians to degrees
  angle_degrees = math.degrees(angle_radians)

  return angle_degrees


def all_models_lexical_stats_plot(
    variations: list,
    show: bool = False,
    models: list = ["gemma3:270m", "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"]
) -> bool:
    """
    Plot the average lexical distances for each model and each variation type.
    Creates a 2x2 grid of subplots for the following distances:
    - Levenshtein
    - Indel
    - Hamming
    - Jaro

    X-axis: variation type
    One line per model in each subplot.
    """
    # Distance types and their pretty names
    distance_types = [
        ("levenshtein_distance", "Levenshtein"),
        ("indel_distance", "Indel"),
        ("hamming_distance", "Hamming"),
        ("jaro_distance", "Jaro"),
    ]
    n_rows, n_cols = 2, 2

    # Prepare colors for models
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    variation_names = [var for var in variations]

    # Prepare data for all models
    model_stats = {}
    for idx, model in enumerate(models):
        model_name_fix = model.replace(':', '_')
        data_file = f"results/{model_name_fix}/lexical_distances_stats.json"
        try:
            stats = dd.load.Json(data_file)
            model_stats[model] = stats
        except Exception:
            model_stats[model] = None

    # Also load the original question stats for reference
    question_stats_file = "results/questions/lexical_distances_stats.json"
    try:
        question_stats = dd.load.Json(question_stats_file)
    except Exception:
        question_stats = None

    # Create the grid of subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10), sharex=True)
    axes = axes.flatten()

    for i, (distance_key, distance_label) in enumerate(distance_types):
        ax = axes[i]
        # Plot each model
        for idx, model in enumerate(models):
            stats = model_stats.get(model)
            if not stats:
                continue
            y_vals = []
            for var in variation_names:
                try:
                    # Use 'variation' distance if available, else fallback to 'question'
                    if "variation" in stats[var]:
                        y_vals.append(stats[var]["variation"][distance_key])
                    else:
                        y_vals.append(stats[var]["question"][distance_key])
                except Exception:
                    y_vals.append(0)
            ax.plot(
                variation_names,
                y_vals,
                marker='o',
                linewidth=2,
                markersize=7,
                color=colors[idx % len(colors)],
                label=model
            )
        # Plot the original question stats as a reference (if available)
        if question_stats:
            y_vals = []
            for var in variation_names:
                try:
                    y_vals.append(question_stats[var]["question"][distance_key])
                except Exception:
                    y_vals.append(0)
            ax.plot(
                variation_names,
                y_vals,
                marker='o',
                linewidth=2,
                markersize=7,
                color='black',
                linestyle='--',
                label='Original Question'
            )
        ax.set_title(f"{distance_label} Distance", fontsize=13, fontweight='bold')
        ax.set_xlabel("Variation Type", fontsize=11)
        ax.set_ylabel(distance_label, fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.suptitle("Average Lexical Distances by Variation Type for Each Model", fontsize=16, fontweight='bold')
    save_file = "results/questions/plots/all_models_lexical_distances_grid.png"
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return True
    

def all_models_word_count_plot(
    variations: list,
    show: bool = False,
    variable_type: str = "character_count",
    models: list = ["gemma3:270m", "gemma3:1b", "gemma3:4b", "gemma3:12b", "gemma3:27b"]) -> bool:
    """
    Plot the word count for each question variation for each model in a single line plot.
    """

    save_file = f"results/questions/plots/all_models_{variable_type}_plot.png"

    # Prepare the plot
    plt.figure(figsize=(8, 6))

    # Use consistent colors for each model
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'cyan', 'magenta']
    variation_names = [var for var in variations]

    for idx, model in enumerate(models):
        model_name_fix = model.replace(':', '_')
        data_file = f"results/{model_name_fix}/lexical_distances_stats.json"
        # Load lexical stats for this model
        stats = dd.load.Json(data_file)
        # Get word counts for each variation (skip 'question' as baseline)
        word_counts = []
        for var in variation_names:
            try:
                word_counts.append(stats[var]['counts'][variable_type])
            except Exception:
                word_counts.append(0)
        plt.plot(
            variation_names,
            word_counts,
            marker='o',
            linewidth=2,
            markersize=8,
            color=colors[idx % len(colors)],
            label=model
        )
    
    data_file = f"results/questions/lexical_distances_stats.json"
    stats = dd.load.Json(data_file)
    word_counts = [stats[var]['counts'][variable_type] for var in variation_names]
    plt.plot(variation_names, word_counts, marker='o', linewidth=2, 
            markersize=8, color='black', linestyle='--', label='Original Variation')

    plt.title(f"{variable_type.replace('_', ' ').title()} by Variation Type for Each Model", fontsize=14, fontweight='bold')
    plt.xlabel("Variation Type", fontsize=12)
    plt.ylabel(f"{variable_type.replace('_', ' ').title()}", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Model", fontsize=10, title_fontsize=11)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()
    return True


def directions_plot(
    variations: list,
    model_name: str,
    data_file: str = "results/questions/questions_embeddings_distances_stats.json",
    save_file: str = "results/questions/plots/questions_embeddings_distances_directions.png",
    show: bool = False,
    relative: bool = False) -> bool:
    """
    """
    angle_type = "angle_degrees"
    distance_type = "euclidean_distance"

    # load statistics
    stats = dd.load.Json(data_file)

    q_distance = 0
    if relative:
        # Get angle in degrees and convert to radians for plotting
        angle_deg = stats["question"][angle_type]["mean"]
        q_angle_rad = np.radians(angle_deg)
        
        # Get distance (length of arrow) using euclidean distance
        q_distance = stats["question"][distance_type]["mean"]

        # Calculate arrow end point
        q_end_x = q_distance * np.cos(q_angle_rad)
        q_end_y = q_distance * np.sin(q_angle_rad)

    center_x, center_y = 0, 0
    
    variation_labels = [variation.replace("_change", "").capitalize() for variation in variations]

    color = "grey"
    shapes = ["o", "s", "d", "v", "p"]
    marker_size = 20

    # Create the arrow plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_box_aspect(1)

    # Plot the center point
    ax.plot(center_x, center_y, 'ko', markersize=marker_size, label='Original Question')

    # Arrow data
    arrows_data = []
    for i, variation in enumerate(variations):
        # Get angle in degrees and convert to radians for plotting
        angle_deg = stats[variation][angle_type]["mean"]

        angle_rad = np.radians(angle_deg) if angle_deg not in [None, 0] else 0
        
        # Get distance (length of arrow) using euclidean distance
        t_distance = stats[variation][distance_type]["mean"] if stats[variation][distance_type]["mean"] not in [None, 0] else 0

        abs_end_x = t_distance * np.cos(angle_rad) 
        abs_end_y = t_distance * np.sin(angle_rad)

        t_angle_rad = calculate_angle_x((q_end_x, q_end_y), (abs_end_x, abs_end_y))
        
        t_distance = t_distance - q_distance
        
        # Calculate arrow end point
        end_x = t_distance * np.cos(t_angle_rad)
        end_y = t_distance * np.sin(t_angle_rad)
        
        # Store arrow data for legend
        arrows_data.append({
            'variation': variation,
            'label': variation_labels[i],
            'color': color,
            'angle': angle_deg,
            'distance': t_distance,
            'x_end': end_x,
            'y_end': end_y
        })
        
        # Draw the marker
        ax.plot(end_x, end_y, marker=shapes[i+1], markersize=marker_size, color=color, 
                alpha=0.8, markeredgecolor='black', markeredgewidth=1)

    # Set equal aspect ratio and limits
    max_x_distance = max([abs(data['x_end']) for data in arrows_data])
    max_y_distance = max([abs(data['y_end']) for data in arrows_data])

    # Calculate plot limits based on center position and max distance
    padding = 1.2
    ax.set_xlim(-max_x_distance*padding, max_x_distance*padding)
    ax.set_ylim(-max_y_distance*padding, max_y_distance*padding)

    # Add grid
    ax.grid(True, alpha=0.3)
    ax.axhline(y=center_y, color='k', linewidth=0.5)
    ax.axvline(x=center_x, color='k', linewidth=0.5)

    # Add angle reference lines for all quadrants from the center point
    for angle in range(0, 360, 30):  # Every 30 degrees from 0 to 330
        angle_rad = np.radians(angle)
        x_end = center_x + max_x_distance * 0.9 * np.cos(angle_rad)
        y_end = center_y + max_y_distance * 0.9 * np.sin(angle_rad)
        ax.plot([center_x, x_end], [center_y, y_end], 'k--', alpha=0.2, linewidth=0.5)
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
    ax.set_xlabel(distance_type.replace('_', ' ').title(), fontsize=12)
    ax.set_ylabel(distance_type.replace('_', ' ').title(), fontsize=12)
    ax.set_title(model_name, fontsize=14, pad=10)

    plt.tight_layout()
    # plt.subplots_adjust(left=0.15, right=0.85, top=0.85, bottom=0.15)
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()

    return True


def stats_boxplot(
    model_name: str,
    variations: list,
    variable_type: str,
    data_file: str,
    save_file: str,
    show: bool = False) -> bool:
    """
    Create a boxplot using statistical summary data (min, Q1, median, Q3, max)
    """

    # load statistics data
    stats_data = dd.load.Json(data_file)

    # Create boxplot visualization, except for the "question"
    plt.figure(figsize=(6, 6))
    
    # Prepare data for boxplot using statistical summaries
    boxplot_data = []
    means = []

    for variation in variations:
        data = []
        for element in stats_data:
            for answer in element[variation]:
                data.append(answer[variable_type])
        boxplot_data.append(data)
        means.append(np.mean(data))

    plt.boxplot(
        boxplot_data,
        labels=variations,
        patch_artist=True,
        boxprops=dict(facecolor='grey', alpha=0.7),
        medianprops=dict(color='red', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5)
    )
    
    # Add mean points
    x_positions = range(1, len(variations) + 1)
    plt.scatter(x_positions, means, color='darkblue', s=100, 
                marker='o', label='Mean', zorder=5)
    plt.ylim(0, 0.65)
    plt.ylabel(variable_type.replace('_', ' ').title(), fontsize=12)
    plt.title(model_name, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    return True


def questions_lexical_stats_plot(
    variations: list,
    comparison: str,
    data_file: str,
    save_file: str,
    show: bool = False) -> bool:
    """
    Create line plots for lexical statistics data.
    Left plot: normalized distance metrics
    Right plot: word count and character count (not normalized)
    """

    # load statistics
    questions_lexical_stats = dd.load.Json(data_file)
    
    # Extract variation names (excluding 'question' as it's the baseline)
    variation_names = [var for var in variations if var != 'question']
    
    # Create figure with two subplots side by side
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    
    # Plot 1: Normalized Distance Metrics (Left)
    distance_metrics = ['levenshtein_distance', 'indel_distance', 'hamming_distance', 
                       'jaro_distance']
    
    # Calculate normalization factors (max values across all variations)
    max_values = {}
    for metric in distance_metrics:
        max_values[metric] = max([questions_lexical_stats[var][comparison][metric] 
                                 for var in variation_names])
    
    # Plot normalized distance metrics
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, metric in enumerate(distance_metrics):
        values = []
        for var in variation_names:
            normalized_value = questions_lexical_stats[var][comparison][metric] / max_values[metric] if max_values[metric] != 0 else 0
            values.append(normalized_value)
        
        ax1.plot(variation_names, values, marker='o', linewidth=2, 
                markersize=8, color=colors[i], label=metric.replace('_', ' ').title())
    
    comparison_label = "Original Question" if comparison == "question" else "Variation"
    ax1.set_title(f'Normalized Distance Metrics by Variation Type, compared to {comparison_label}', fontsize=14, fontweight='bold')
    
    # Add legend inside the plot at bottom right
    ax1.legend(ncol=1, bbox_to_anchor=(0.80, 0.05), loc='lower right', 
              frameon=True, fancybox=True, shadow=False)
    
    ax1.set_ylabel('Normalized Value (0-1)', fontsize=12)
    ax1.set_xlabel('Variation Type', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Rotate x-axis labels for better readability
    ax1.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()
    
    return True


def word_count_plot(
    variations: list,
    show: bool = False,
    data_file: str = "results/questions/questions_lexical_distances_stats.json",
    save_file: str = "results/questions/plots/questions_word_count_plot.png") -> bool:
    """
    Create line plots for lexical statistics data.
    Left plot: normalized distance metrics
    Right plot: word count and character count (not normalized)
    """

    # load statistics
    questions_lexical_stats = dd.load.Json(data_file)
    
    # Extract variation names (excluding 'question' as it's the baseline)
    variation_names = [var for var in variations if var != 'question']
    
    # Create figure with two subplots side by side
    fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))
        
    # Word Count and Character Count
    word_counts = [questions_lexical_stats[var]['counts']['word_count'] for var in variation_names]
    char_counts = [questions_lexical_stats[var]['counts']['character_count'] for var in variation_names]
    
    ax.plot(variation_names, word_counts, marker='s', linewidth=2, 
            markersize=8, color='darkblue', label='Word Count')
    ax.plot(variation_names, char_counts, marker='^', linewidth=2, 
            markersize=8, color='darkred', label='Character Count')
    
    ax.set_title('Word Count and Character Count by Variation Type', fontsize=14, fontweight='bold')
    
    # Add legend inside the plot, moved right and up
    ax.legend(ncol=1, bbox_to_anchor=(0.05, 0.95), loc='upper left', 
              frameon=True, fancybox=True, shadow=False)
    
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xlabel('Variation Type', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()
    
    return True


