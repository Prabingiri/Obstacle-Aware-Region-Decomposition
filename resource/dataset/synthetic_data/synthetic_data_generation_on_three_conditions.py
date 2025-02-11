import os
import random
import math
import json
import logging
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union

logging.basicConfig(level=logging.INFO)
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon


class ResearchDataGenerator:
    def __init__(self, region_size=(100, 100), obstacle_percentage=10, max_obstacle_size=None, size_variation=0.5, seed=None):
        """
        Initializes the synthetic data generator.

        Parameters:
        - region_size (tuple): Size of the region (width, height).
        - obstacle_percentage (float): Total area of obstacles as a percentage of the region.
        - max_obstacle_size (float): Maximum area of a single obstacle.
        - size_variation (float): Variation in obstacle sizes (0 to 1, where 1 is high variation).
        - seed (int): Random seed for reproducibility.
        """
        self.region_size = region_size
        self.obstacle_percentage = obstacle_percentage
        self.max_obstacle_size = max_obstacle_size
        self.size_variation = size_variation
        if seed is not None:
            random.seed(seed)

    def generate_region(self):
        """Generates a rectangular region."""
        width, height = self.region_size
        return Polygon([(0, 0), (width, 0), (width, height), (0, height), (0, 0)])

    def generate_convex_polygon(self, center, size, sides):
        """Generates a random convex polygon."""
        angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(sides))
        points = [
            (
                center[0] + size * math.cos(angle),
                center[1] + size * math.sin(angle)
            )
            for angle in angles
        ]
        return Polygon(points).convex_hull

    def generate_obstacles(self, region):
        """Generates non-overlapping, simple convex obstacles based on the specified parameters."""
        width, height = self.region_size
        max_total_area = (self.obstacle_percentage / 100) * region.area
        obstacles = []
        current_area = 0

        while current_area < max_total_area:
            # Generate obstacle parameters
            center = (random.uniform(0, width), random.uniform(0, height))
            base_size = max_total_area / 50  # Base size relative to total area
            size = base_size * (1 + self.size_variation * random.uniform(-1, 1))

            # Respect max_obstacle_size
            if self.max_obstacle_size:
                max_size_limit = math.sqrt(self.max_obstacle_size / math.pi)
                size = min(size, max_size_limit)

            sides = random.choice([3, 4, 5, 6])  # Triangles, rectangles, pentagons, hexagons
            obstacle = self.generate_convex_polygon(center, size, sides)

            # Ensure obstacle is valid, convex, and non-overlapping
            if (
                region.contains(obstacle) and
                obstacle.is_valid and
                all(not obstacle.intersects(existing) for existing in obstacles)
            ):
                obstacles.append(obstacle)
                current_area += obstacle.area

        return obstacles

    def ensure_connectivity(self, region, obstacles):
        """Ensures the region remains connected after adding obstacles."""
        valid_obstacles = []
        for obstacle in obstacles:
            temp_region = region.difference(obstacle)
            if temp_region.geom_type == "Polygon":
                valid_obstacles.append(obstacle)
            else:
                logging.warning("Obstacle discarded to maintain region connectivity.")
        return valid_obstacles

    def generate_and_store(self, file_path):
        """Generates the region and obstacles and stores them in a JSON file."""
        region = self.generate_region()
        obstacles = self.generate_obstacles(region)
        obstacles = self.ensure_connectivity(region, obstacles)

        data = {"region": mapping(region), "obstacles": [mapping(obstacle) for obstacle in obstacles]}

        with open(file_path, 'w') as f:
            json.dump(data, f)
        logging.info(f"Synthetic data saved to {file_path}.")

    @staticmethod
    def load_from_file(file_path):
        """Loads synthetic data from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)

        region = shape(data["region"])
        obstacles = [shape(obstacle) for obstacle in data["obstacles"]]
        return region, obstacles



if __name__ == "__main__":
    # Parameters for experimentation
    region_size = (100, 100)
    obstacle_percentage = 5
    max_obstacle_size = 50
    size_variation = 1
    seed = 40

    generator = ResearchDataGenerator(
        region_size=region_size,
        obstacle_percentage=obstacle_percentage,
        max_obstacle_size=max_obstacle_size,
        size_variation=size_variation,
        seed=seed
    )

    # Define directory and file naming
    region_dir = f"synthetic_data_generated/{region_size[0]}x{region_size[1]}"
    # Ensure the directory exists
    file_name = f"synthetic_data_{obstacle_percentage}percent_{max_obstacle_size}maxobs_{size_variation}var.json"

    file_path = os.path.join(region_dir, file_name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Generate and save data
    generator.generate_and_store(file_path)

    # Load and analyze data
    region, obstacles = ResearchDataGenerator.load_from_file(file_path)
    print(f"Generated {len(obstacles)} obstacles with total area {sum(o.area for o in obstacles):.2f}")