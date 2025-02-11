import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np


def visualize_strip(region, obstacles, strip, xj_minus_1, xj, perimeter, accumulated_perimeter, strip_index):
    """
    Visualize a single strip with contributing obstacles and perimeter details.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot the region
    region_patch = MplPolygon(np.array(region.exterior.coords), closed=True, edgecolor='black', facecolor='none', lw=2)
    ax.add_patch(region_patch)

    # Plot the strip
    strip_patch = MplPolygon(np.array(strip.exterior.coords), closed=True, edgecolor='red', facecolor='none', linestyle="--", lw=2)
    ax.add_patch(strip_patch)

    # Plot the obstacles
    for obstacle in obstacles:
        coords = np.array(obstacle.exterior.coords)
        ax.add_patch(MplPolygon(coords, closed=True, edgecolor='blue', facecolor='cyan', alpha=0.5, lw=1))

    # Display perimeter information
    ax.text((xj_minus_1 + xj) / 2, region.bounds[3] + 2,
            f"Strip {strip_index}\nPerimeter: {perimeter:.2f}\nAccumulated: {accumulated_perimeter:.2f}",
            fontsize=10, color='red', ha='center')

    # Plot settings
    ax.set_xlim(region.bounds[0] - 5, region.bounds[2] + 5)
    ax.set_ylim(region.bounds[1] - 5, region.bounds[3] + 5)
    ax.set_aspect('equal')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.title(f"Strip [{xj_minus_1}, {xj}] and Obstacles")
    plt.show()


def visualize_all_strips(region, obstacles, strips, perimeters, cumulative_perimeters):
    """
    Visualize all strips sequentially.
    """
    for i, (xj_minus_1, xj, strip) in enumerate(strips, start=1):
        perimeter = perimeters.get((xj_minus_1, xj), 0.0)
        accumulated_perimeter = cumulative_perimeters.get(xj, 0.0)
        visualize_strip(region, obstacles, strip, xj_minus_1, xj, perimeter, accumulated_perimeter, i)
