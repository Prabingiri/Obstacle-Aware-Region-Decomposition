import logging
import random
import matplotlib.pyplot as plt
from shapely import make_valid
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.plotting import plot_polygon
from shapely.ops import unary_union

from src.strip_perimeter import Strip
from src.obstacle_aware_divider import ObstacleAwareDivider

DRONE_THRESHOLD = 5.0
COVERAGE_RATIO_STOP = 0.90

class NaiveKDTreePartitioning:
    """
    Naive KD-Tree approach. Instead of using half-perimeter,
    we simply split at the bounding box midpoint along the chosen axis.
    Everything else (obstacle clipping, stopping conditions) is the same
    as KDTreePartitioning so we can compare fairly.
    """

    def __init__(self, region, obstacles, max_depth, min_area_threshold=1e-3,
                 advanced_checks=False, check_connectivity=False):
        self.region, self.obstacles = validate_and_fix_geometries(region, obstacles)
        self.max_depth = max_depth
        self.min_area_threshold = min_area_threshold
        self.advanced_checks = advanced_checks
        self.check_connectivity = check_connectivity
        self.partitions = []

    def naive_kd_partition(self, region, obstacles, depth, axis='x'):
        if depth <= 0 or region.area < self.min_area_threshold:
            logging.info(f"Stopping partitioning (naive): Depth={depth}, Area={region.area:.2f}")
            self.partitions.append((region, obstacles))
            return

        region = validate_geometry(region)
        if not region:
            logging.warning("Region invalid or degenerate => stopping.")
            self.partitions.append((region, obstacles))
            return

        if self.advanced_checks:
            # If advanced checks are enabled => coverage & leftover checks
            if self._check_coverage_and_stop(region, obstacles):
                # Store region if it meets coverage≥90% & leftover<DRONE etc.
                self.partitions.append((region, obstacles))
                return

        minx, miny, maxx, maxy = region.bounds
        # The only difference: we choose the bounding box midpoint
        if axis == 'x':
            division_point = 0.5 * (minx + maxx)
        else:
            division_point = 0.5 * (miny + maxy)

        # Divide region using bounding boxes
        if axis == 'x':
            left_box = box(minx, miny, division_point, maxy)
            right_box = box(division_point, miny, maxx, maxy)
        else:
            left_box = box(minx, miny, maxx, division_point)
            right_box = box(minx, division_point, maxx, maxy)

        # If sub-box is empty => store parent
        if left_box.is_empty or right_box.is_empty:
            logging.warning("One sub-box empty => store parent region.")
            self.partitions.append((region, obstacles))
            return

        # Clip obstacles
        left_obs = []
        right_obs = []
        for ob in obstacles:
            inter_left = ob.intersection(left_box)
            if not inter_left.is_empty:
                left_obs.append(make_valid(inter_left))

            inter_right = ob.intersection(right_box)
            if not inter_right.is_empty:
                right_obs.append(make_valid(inter_right))

        # Validate sub-box dimension
        left_box = validate_geometry(left_box.intersection(region))
        right_box = validate_geometry(right_box.intersection(region))

        # If advanced checks => dimension threshold & connectivity
        if self.advanced_checks:
            if not self._is_subregion_valid(left_box, left_obs):
                left_box = None
            if not self._is_subregion_valid(right_box, right_obs):
                right_box = None


        if left_box:
            lxmin, lymin, lxmax, lymax = left_box.bounds
            if (lxmax - lxmin < self.min_area_threshold) or (lymax - lymin < self.min_area_threshold):
                left_box = None

        if right_box:
            rxmin, rymin, rxmax, rymax = right_box.bounds
            if (rxmax - rxmin < self.min_area_threshold) or (rymax - rymin < self.min_area_threshold):
                right_box = None

        if not left_box and not right_box:
            logging.warning("Both sub-boxes invalid => store parent region (naive).")
            self.partitions.append((region, obstacles))
            return

        next_axis = 'y' if axis=='x' else 'x'
        if left_box:
            self.naive_kd_partition(left_box, left_obs, depth-1, next_axis)
        if right_box:
            self.naive_kd_partition(right_box, right_obs, depth-1, next_axis)

    def run(self):
        """Initiate naive KD-Tree partitioning."""
        self.naive_kd_partition(self.region, self.obstacles, self.max_depth, axis='x')
        logging.info(f"Naive KD-Tree partitioning complete. Total partitions: {len(self.partitions)}")
        return self.partitions

    # ------------------------------------
    #   ADVANCED CHECKS (optional)
    # ------------------------------------
    def _check_coverage_and_stop(self, region, obstacles):
        """
        If coverage≥90% & leftover < DRONE_THRESHOLD => store region.
        Returns True if we want to store (stop), else False => keep splitting.
        """
        region_area = region.area
        obs_area = sum(ob.area for ob in obstacles)
        coverage_ratio = obs_area / region_area if region_area > 1e-12 else 1.0
        free_area = region_area - obs_area

        if coverage_ratio >= COVERAGE_RATIO_STOP:
            largest_hole_area = self._compute_largest_free_space(region, obstacles)
            if free_area < DRONE_THRESHOLD and largest_hole_area < DRONE_THRESHOLD:
                return True
        return False

    def _is_subregion_valid(self, region, obstacles):
        """
        Checks dimension threshold, coverage, connectivity if advanced_checks=True.
        """
        if not region or region.is_empty:
            return False

        minx, miny, maxx, maxy = region.bounds
        width = maxx - minx
        height = maxy - miny
        if width < self.min_area_threshold or height < self.min_area_threshold:
            return False

        # Check if obstacles cover entire subregion
        region_area = region.area
        obs_area_sum = sum(ob.area for ob in obstacles)
        if obs_area_sum >= region_area - 1e-9:
            return False

        # Optional connectivity check
        if self.check_connectivity:
            largest_hole = self._compute_largest_free_space(region, obstacles)
            if largest_hole < DRONE_THRESHOLD:
                return False

        return True

    def _compute_largest_free_space(self, region, obstacles):
        """
        If advanced checks => measure largest hole in region after subtracting obstacles.
        """
        try:
            union_obs = unary_union(obstacles)
            free_space = region.difference(union_obs)
            if free_space.is_empty:
                return 0.0
            if isinstance(free_space, Polygon):
                return free_space.area
            elif isinstance(free_space, MultiPolygon):
                return max(poly.area for poly in free_space.geoms)
            else:
                max_area = 0.0
                for g in free_space.geoms:
                    if g.geom_type in ("Polygon", "MultiPolygon"):
                        max_area = max(max_area, g.area)
                return max_area
        except:
            return 0.0


    def visualize_partitions(self, datatype, filename=None, show_obstacles=True, show_wcrt=True):
        """
        Visualization of final partitions using matplotlib.
        Saves the plot to the given filename if provided.
        """
        import matplotlib.pyplot as plt
        from shapely.plotting import plot_polygon

        fig, ax = plt.subplots(figsize=(12, 10))
        colors = [tuple(random.random() for _ in range(3)) for _ in range(len(self.partitions))]

        for i, (subregion, subobs) in enumerate(self.partitions):
            if not subregion or subregion.is_empty or subregion.geom_type not in {"Polygon", "MultiPolygon"}:
                continue

            if subregion.geom_type == "MultiPolygon":
                geoms = subregion.geoms
            else:
                geoms = [subregion]

            c = colors[i]
            for g in geoms:
                plot_polygon(g, ax=ax, facecolor=c, alpha=0.4, edgecolor="black")

            cx, cy = subregion.centroid.x, subregion.centroid.y
            ax.text(cx, cy, f"Partition {i + 1}", ha="center", va="center", fontsize=8)

            if show_obstacles:
                for ob in subobs:
                    if ob.geom_type in {"Polygon", "MultiPolygon"}:
                        plot_polygon(ob, ax=ax, facecolor="none", edgecolor="red")

        ax.set_aspect("equal")

        # Determine filename if not provided
        if filename is None:
            filename = f"naive_kd_tree_partitions_{datatype}.png"

        plt.savefig(filename, dpi=300)
        plt.close(fig)  # Close the figure after saving
        print(f"Partition visualization saved to {filename}")

    def save_partitions(self, datatype, depth, output_dir=f"results/naive_KD_tree", runtime=None, strategies_data=None,
                        percentage=None):
        """
        Save the partition information into a CSV file and optionally update the strategies_data dictionary.

        Parameters
        ----------
        datatype : str
            Describes the type of data being partitioned (e.g., synthetic, real-world).
        depth : int
            The maximum depth of the KD-Tree partitioning.
        output_dir : str
            Directory to save the CSV file and text tree.
        runtime : float
            The execution time for the partitioning process.
        strategies_data : dict
            Dictionary to store the max WCRT and standard deviation values for later analysis.
        percentage : int
            Obstacle percentage for the current data.
        """
        import os
        import csv
        import json
        from datetime import datetime

        if datatype != "iowa" and datatype != "Synthetic":
            obstacle_percent = f"obstacle_{percentage}" if percentage is not None else "unknown"
            output_dir = f"{output_dir}/{datatype}/{obstacle_percent}/{depth}"
        else:
            output_dir = f"{output_dir}/{datatype}/{depth}"

        if self.advanced_checks:
            output_dir = f"{output_dir}_advanced_checks"

        # Create output directory if not exists
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(output_dir, f"{datatype}_naive_kd_tree_depth_{depth}_{timestamp}.csv")
        tree_filename = os.path.join(output_dir, f"{datatype}_naive_kd_tree_depth_{depth}_{timestamp}_tree.txt")
        visualization_filename = os.path.splitext(csv_filename)[0] + "_partition.png"

        # Directory-independent summary file
        summary_file = "results/final/Naive-kd-tree/summary_data.json"
        os.makedirs("results/final/Naive-kd-tree/", exist_ok=True)

        # Initialize summary data if not present
        if not os.path.exists(summary_file):
            summary_data = {}
        else:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)

        # Parse percentage from datatype
        percentage = f"{datatype.split('_')[-1]}%" if "percent" in datatype else "N/A"
        # Ensure the structure for this percentage exists

        # CSV file generation for partition data
        # now_str = datetime.datetime.now().strftime("%Y%m%d")  # Only date

        if percentage not in summary_data:
            summary_data[percentage] = {"max_wcrt": [], "std_dev": []}

        # Define CSV columns
        columns = [
            "datatype",
            "partition_number",
            "num_obstacles",
            "WCRT",
            "aspect_ratio",
            "min_wcrt",
            "max_wcrt",
            "variance",
            "standard_deviation",
            "range",
            "runtime",
            "partition_boundary",
            "obstacles_in_partition"
        ]

        # Collect WCRT and aspect ratio values for global stats
        wcrt_values = []
        aspect_ratios = []

        # Write CSV
        with open(csv_filename, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for i, (subregion, subobs) in enumerate(self.partitions, start=1):
                if subregion.is_empty or subregion.geom_type not in {"Polygon", "MultiPolygon"}:
                    logging.warning(f"Skipping invalid or non-polygon geometry in partition {i}: {subregion.geom_type}")
                    continue

                # WCRT calculation
                strip_mgr = Strip(subregion, subobs)
                wcrt_val = strip_mgr.calculate_region_wcrt()
                wcrt_values.append(wcrt_val)

                # Aspect ratio calculation
                aspect_ratio_val = compute_aspect_ratio(subregion)
                aspect_ratios.append(aspect_ratio_val)

                # Partition boundary and obstacles
                partition_boundary_str = subregion.wkt
                obstacles_str = ";".join([f"{o.bounds}" for o in subobs])

                writer.writerow({
                    "datatype": datatype,
                    "partition_number": i,
                    "num_obstacles": len(subobs),
                    "WCRT": round(wcrt_val, 2),
                    "aspect_ratio": round(aspect_ratio_val, 3),
                    "min_wcrt": "",
                    "max_wcrt": "",
                    "variance": "",
                    "standard_deviation": "",
                    "range": "",
                    "runtime": "",
                    "partition_boundary": partition_boundary_str,
                    "obstacles_in_partition": obstacles_str
                })

        # Calculate global stats
        min_wcrt = min(wcrt_values)
        max_wcrt = max(wcrt_values)
        range_wcrt = max_wcrt - min_wcrt
        average_wcrt = sum(wcrt_values) / len(wcrt_values) if wcrt_values else 0
        variance_wcrt = sum((w - average_wcrt) ** 2 for w in wcrt_values) / len(wcrt_values) if wcrt_values else 0
        std_dev_wcrt = variance_wcrt ** 0.5

        # Save global stats row
        with open(csv_filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writerow({
                "datatype": datatype,
                "partition_number": len(self.partitions),
                "num_obstacles": "N/A",
                "WCRT": round(average_wcrt, 2),
                "aspect_ratio": round(sum(aspect_ratios) / len(aspect_ratios), 3) if aspect_ratios else 0,
                "min_wcrt": round(min_wcrt, 2),
                "max_wcrt": round(max_wcrt, 2),
                "variance": round(variance_wcrt, 3),
                "standard_deviation": round(std_dev_wcrt, 3),
                "range": round(range_wcrt, 2),
                "runtime": runtime,
                "partition_boundary": "N/A",
                "obstacles_in_partition": "N/A"
            })

        logging.info(f"Partition data saved to '{csv_filename}'.")

        # Extract WCRT and std_dev from the last row
        summary_data[percentage]["max_wcrt"].append(round(max_wcrt, 2))
        summary_data[percentage]["std_dev"].append(round(std_dev_wcrt, 3))

        # Save updated summary data
        with open(summary_file, "w") as f:
            json.dump(summary_data, f, indent=4)

        logging.info(f"Summary data updated for {percentage}, depth={depth}.")

        # Create a text tree representation
        with open(tree_filename, "w", encoding="utf-8") as tf:
            for i, (subregion, subobs) in enumerate(self.partitions, start=1):
                tf.write(
                    f"Partition {i}: Area={round(subregion.area, 2)}, WCRT={round(wcrt_values[i - 1], 2)}, "
                    f"#Obstacles={len(subobs)}, Bounds={subregion.bounds}\n"
                )

        logging.info(f"Partition tree saved to '{tree_filename}'.")

        # Save visualization
        try:
            self.visualize_partitions(datatype, filename=visualization_filename,
                                      show_wcrt=True, show_obstacles=True)
        except Exception as e:
            logging.error(f"Visualization failed: {e}")


def validate_and_fix_geometries(region, obstacles):
    """
    Validates and fixes geometries for the region and obstacles.

    Args:
        region (Polygon): The region polygon.
        obstacles (list[Polygon]): List of obstacle polygons.

    Returns:
        tuple: Fixed region polygon and list of fixed obstacle polygons.
    """
    if not region.is_valid:
        region = region.buffer(0)
    fixed_obstacles = [obs.buffer(0) if not obs.is_valid else obs for obs in obstacles]
    return region, fixed_obstacles

def validate_geometry(geometry):
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    if geometry.is_empty or not isinstance(geometry, (Polygon, MultiPolygon)):
        return None  # Skip invalid or empty geometries
    return geometry

def compute_aspect_ratio(geometry, eps=1e-9):
    """
    Compute squareness as the minimum of w/h and h/w.
    """
    minx, miny, maxx, maxy = geometry.bounds
    w = maxx - minx
    h = maxy - miny
    if abs(w) < eps or abs(h) < eps:
        return 1.0  # Degenerate shapes are treated as perfectly square
    return min(w / h, h / w)
