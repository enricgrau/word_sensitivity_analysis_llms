import datadocket as dd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

# params
VARIATIONS = ["synonym_change", "antonym_change", "paraphrase_change", "letter_change"]
VARIATION_LABELS = ["Synonym\nChange", "Antonym\nChange", "Paraphrase\nChange", "Letter\nChange"]
COLORS = ['lightcoral', 'lightgreen', 'gold', 'plum']

# load statistics
directions_stats = dd.load.Json("data/directions_stats.json")
distances_stats = dd.load.Json("data/distances_stats.json")

# Create the arrow plot
fig, ax = plt.subplots(figsize=(12, 12))

# Center point (origin)
center_x, center_y = 0, 0

# Plot the center point
ax.plot(center_x, center_y, 'ko', markersize=15, label='Original Question')

# Arrow data
arrows_data = []

for i, variation in enumerate(VARIATIONS):
    # Get angle in degrees and convert to radians for plotting
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
        'label': VARIATION_LABELS[i],
        'color': COLORS[i],
        'angle': angle_deg,
        'distance': distance
    })
    
    # Draw the arrow
    ax.annotate('', xy=(end_x, end_y), xytext=(center_x, center_y),
                arrowprops=dict(arrowstyle='->', lw=3, color=COLORS[i], alpha=0.8))
    
    # Add label at the end of the arrow
    ax.annotate(VARIATION_LABELS[i], xy=(end_x, end_y), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=10, ha='left', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS[i], alpha=0.7))

# Set equal aspect ratio and limits
max_distance = max([data['distance'] for data in arrows_data])
ax.set_xlim(-max_distance * 1.2, max_distance * 1.2)
ax.set_ylim(-max_distance * 1.2, max_distance * 1.2)
ax.set_aspect('equal')

# Add grid
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)

# Add angle reference lines (optional)
for angle in [0, 30, 60, 90, 120, 150, 180]:
    angle_rad = np.radians(angle)
    x_end = max_distance * 1.1 * np.cos(angle_rad)
    y_end = max_distance * 1.1 * np.sin(angle_rad)
    ax.plot([0, x_end], [0, y_end], 'k--', alpha=0.2, linewidth=0.5)
    ax.text(x_end * 1.05, y_end * 1.05, f'{angle}°', fontsize=8, alpha=0.6)

# Create legend
legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.7, label=label) 
                  for color, label in zip(COLORS, VARIATION_LABELS)]
legend_elements.insert(0, plt.Line2D([0], [0], marker='o', color='k', linestyle='None', 
                                    markersize=10, label='Original Question'))

ax.legend(handles=legend_elements, title="Variation Types", 
          bbox_to_anchor=(1.05, 1), loc='upper left')

# Labels and title
ax.set_xlabel('Distance (Embedding Space)', fontsize=12)
ax.set_ylabel('Distance (Embedding Space)', fontsize=12)
ax.set_title('Question Variation Directions and Magnitudes\n(Arrow Length = Average Distance, Direction = Average Angle)', 
             fontsize=14, pad=20)

# Add statistics text box
stats_text = "Statistics:\n"
for data in arrows_data:
    stats_text += f"{data['label']}: {data['angle']:.1f}°, {data['distance']:.2f}\n"

ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig("plots/arrow_directions.png", bbox_inches='tight', dpi=300)
plt.show()

# Print detailed statistics
print("\nArrow Plot Statistics:")
print("-" * 50)
print(f"{'Variation':<20} {'Angle (°)':<12} {'Distance':<12} {'Direction'}")
print("-" * 50)
for data in arrows_data:
    direction = "Right" if data['angle'] < 90 else "Left" if data['angle'] > 90 else "Up"
    print(f"{data['label']:<20} {data['angle']:<12.1f} {data['distance']:<12.2f} {direction}")