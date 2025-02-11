import os
import random
import math
import json
import logging
import numpy as np
from shapely.geometry import Polygon, shape, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from shapely.plotting import plot_polygon

logging.basicConfig(level=logging.INFO)


class ResearchDataGenerator:
    def __init__(self, region_size=(100, 100), obstacle_percentage=10,
                 max_obstacle_size=None, size_variation=0.5, seed=None,
                 clusters=None):
        self.region_size = region_size
        self.obstacle_percentage = obstacle_percentage
        self.max_obstacle_size = max_obstacle_size
        self.size_variation = size_variation
        self.clusters = clusters or []
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def generate_region(self):
        width, height = self.region_size
        return Polygon([(0, 0), (width, 0), (width, height), (0, height)])

    def generate_convex_polygon(self, center, size, sides):
        angles = sorted(random.uniform(0, 2 * math.pi) for _ in range(sides))
        points = [(center[0] + size * math.cos(angle),
                   center[1] + size * math.sin(angle)) for angle in angles]
        return Polygon(points).convex_hull

    def sample_center(self):
        width, height = self.region_size
        if not self.clusters:
            return (random.uniform(0, width), random.uniform(0, height))

        weights = [cluster['weight'] for cluster in self.clusters]
        probabilities = [w / sum(weights) for w in weights]
        chosen_cluster = np.random.choice(self.clusters, p=probabilities)
        cx, cy = chosen_cluster['center']
        std_dev = chosen_cluster['std_dev']
        x = np.random.normal(cx, std_dev)
        y = np.random.normal(cy, std_dev)
        x = min(max(x, 0), width)
        y = min(max(y, 0), height)
        return (x, y)

    def generate_obstacles(self, region):
        width, height = self.region_size
        max_total_area = (self.obstacle_percentage / 100) * region.area
        obstacles = []
        current_area = 0

        no_progress_limit = 1000
        no_progress_counter = 0
        original_variation = self.size_variation

        while current_area < max_total_area:
            center = self.sample_center()
            base_size = max_total_area / 50
            size = base_size * (1 + self.size_variation * random.uniform(-1, 1))

            if self.max_obstacle_size:
                max_size_limit = math.sqrt(self.max_obstacle_size / math.pi)
                size = min(size, max_size_limit)

            sides = random.choice([3, 4, 5, 6])
            obstacle = self.generate_convex_polygon(center, size, sides)

            if (region.contains(obstacle) and
                    obstacle.is_valid and
                    all(not obstacle.intersects(existing) for existing in obstacles)):
                obstacles.append(obstacle)
                current_area += obstacle.area
                no_progress_counter = 0
            else:
                no_progress_counter += 1

            if no_progress_counter >= no_progress_limit:
                if self.size_variation > 0.1:
                    self.size_variation = max(0.1, self.size_variation - 0.1)
                    logging.info(f"Reducing size variation to {self.size_variation}.")
                    no_progress_counter = 0
                else:
                    logging.warning("No further progress; stopping generation.")
                    break

        if current_area < max_total_area:
            logging.warning(
                f"Target area not fully reached. Target: {max_total_area:.2f}, Achieved: {current_area:.2f}")

        self.size_variation = original_variation
        return obstacles

    def ensure_connectivity(self, region, obstacles):
        valid_obstacles = []
        for obstacle in obstacles:
            temp_region = region.difference(obstacle)
            if temp_region.geom_type in ["Polygon", "MultiPolygon"]:
                valid_obstacles.append(obstacle)
            else:
                logging.warning("Obstacle discarded to maintain connectivity.")
        return valid_obstacles

    def generate_and_store(self, file_path):
        region = self.generate_region()
        obstacles = self.generate_obstacles(region)
        obstacles = self.ensure_connectivity(region, obstacles)

        total_area = sum(ob.area for ob in obstacles)
        data = {
            "region": mapping(region),
            "obstacles": [mapping(obstacle) for obstacle in obstacles],
            "total_obstacle_area": total_area,
            "target_obstacle_area": (self.obstacle_percentage / 100) * region.area
        }

        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)

        logging.info(f"Synthetic data saved to {file_path}.")
        logging.info(
            f"Target Area: {data['target_obstacle_area']:.2f}, Achieved Area: {data['total_obstacle_area']:.2f}, Count: {len(obstacles)}")

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        region = shape(data["region"])
        obstacles = [shape(obstacle) for obstacle in data["obstacles"]]
        return region, obstacles, data.get("total_obstacle_area", 0), data.get("target_obstacle_area", 0)

    def visualize_obstacles(self, region, obstacles, title="Obstacle Distribution"):
        fig, ax = plt.subplots(figsize=(8, 8))
        plot_polygon(region, ax=ax, facecolor='none', edgecolor='black', linewidth=2, label='Region')
        for obstacle in obstacles:
            plot_polygon(obstacle, ax=ax, facecolor='red', edgecolor='darkred', alpha=0.6)
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("X Coordinate", fontsize=14)
        ax.set_ylabel("Y Coordinate", fontsize=14)
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# Testing for obstacle percentages from 0% to 25%
if __name__ == "__main__":
    region_size = (100, 100)
    max_obstacle_size = 200
    size_variation = 0.5
    seed = 73

    clusters = [
        {"center": (20, 20), "std_dev": 30, "weight": 10},
        # {"center": (80, 80), "std_dev": 5, "weight": 10},
        # {"center": (80, 80), "std_dev": 10, "weight": 5}
    ]

    for obstacle_percentage in range(0, 35, 5):
        logging.info(f"Generating data for {obstacle_percentage}% obstacle coverage.")
        generator = ResearchDataGenerator(
            region_size=region_size,
            obstacle_percentage=obstacle_percentage,
            max_obstacle_size=max_obstacle_size,
            size_variation=size_variation,
            seed=seed,
            clusters=clusters
        )

        file_path = f"synthetic_data_generation/leftbiased/{obstacle_percentage}_percent_obstacles.json"
        generator.generate_and_store(file_path)

        region, obstacles, total_area, target_area = ResearchDataGenerator.load_from_file(file_path)
        achieved_coverage = (total_area / target_area) * 100 if target_area else 0
        logging.info(f"{obstacle_percentage}% target: Generated {len(obstacles)} obstacles, "
                     f"Total Area: {total_area:.2f}, Achieved Coverage: {achieved_coverage:.2f}%.")
