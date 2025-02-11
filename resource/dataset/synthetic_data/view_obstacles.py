import logging
import os

from synthetic_data_generation_on_three_conditions import ResearchDataGenerator
logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon

def visualize_obstacles(region, obstacles, obstacle_percentage, size_variation, max_obstacle_size, json_file_path):
    """
    Visualizes the region and obstacles using matplotlib and saves the plot in the same directory as the JSON file.

    Parameters:
    - region: The rectangular region as a Polygon.
    - obstacles: List of obstacle Polygons.
    - obstacle_percentage: Percentage of the region area covered by obstacles.
    - size_variation: Variation in obstacle sizes.
    - max_obstacle_size: Maximum allowable size of a single obstacle.
    - json_file_path: Path to the JSON file where the data is stored.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon
    import os

    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot region boundary
    region_coords = list(region.exterior.coords)
    ax.plot(*zip(*region_coords), color="black", linewidth=2, label="Region Boundary")

    # Plot obstacles
    for obstacle in obstacles:
        obstacle_coords = list(obstacle.exterior.coords)
        ax.add_patch(
            MplPolygon(obstacle_coords, closed=True, edgecolor="red", facecolor="blue", alpha=0.5)
        )

    # Add title with parameters
    title = (
        f"Obstacle Visualization\n"
        f"Coverage: {obstacle_percentage}% | Max Size: {max_obstacle_size} | Size Variation: {size_variation}"
    )
    ax.set_title(title)

    ax.set_xlim(0, region.bounds[2])
    ax.set_ylim(0, region.bounds[3])
    ax.set_aspect("equal", adjustable="box")
    plt.legend()

    # Save the figure
    json_dir = os.path.dirname(json_file_path)
    image_file_name = os.path.splitext(os.path.basename(json_file_path))[0] + ".png"
    image_file_path = os.path.join(json_dir, image_file_name)
    plt.savefig(image_file_path)
    print(f"Visualization saved at: {image_file_path}")

    plt.show()


# Example usage
# Define directory and file naming dynamically
region_size = (100, 100)
obstacle_percentage = 5
max_obstacle_size = 50
size_variation = 1
seed = 40

# Define directory and file naming
region_dir = f"synthetic_data_generated/{region_size[0]}x{region_size[1]}"
file_name = f"synthetic_data_{obstacle_percentage}percent_{max_obstacle_size}maxobs_{size_variation}var.json"
file_path = os.path.join(region_dir, file_name)

# Verify file existence before loading
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    # Load and analyze data
    region, obstacles = ResearchDataGenerator.load_from_file(file_path)

    # Visualize obstacles
    # Visualize and save
    visualize_obstacles(
        region=region,
        obstacles=obstacles,
        obstacle_percentage=obstacle_percentage,
        size_variation=size_variation,
        max_obstacle_size=max_obstacle_size,
        json_file_path=file_path
    )
