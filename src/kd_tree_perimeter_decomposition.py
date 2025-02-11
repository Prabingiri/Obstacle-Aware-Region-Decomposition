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


class KDTreePartitioning:
    def __init__(self, region, obstacles, max_depth, min_area_threshold=1e-3,
                 advanced_checks=False, check_connectivity=False):
        self.region, self.obstacles = validate_and_fix_geometries(region, obstacles)
        self.max_depth = max_depth
        self.min_area_threshold = min_area_threshold
        self.advanced_checks = advanced_checks
        self.check_connectivity = check_connectivity
        self.partitions = []

    def kd_tree_partition(self, region, obstacles, depth, axis='x'):
        if depth <= 0 or region.area < self.min_area_threshold:
            logging.info(f"Stopping partitioning: Depth={depth}, Area={region.area}")
            self.partitions.append((region, obstacles))
            return

        # Validate region
        region = validate_geometry(region)
        if not region:
            logging.warning("Region is invalid or degenerate. Stopping.")
            self.partitions.append((region, obstacles))
            return

        # If advanced checks => coverage≥90% & leftover < DRONE => store
        if self.advanced_checks:
            if self._check_coverage_and_stop(region, obstacles):
                self.partitions.append((region, obstacles))
                return

        # Use Strip to calculate total perimeter and cumulative perimeters
        strip_manager = Strip(region, obstacles, axis=axis)

        total_perimeter = strip_manager.calculate_total_obstacle_perimeter()
        half_perimeter = total_perimeter / 2

        division_point = None
        for coord_prev, coord_curr, _ in strip_manager.strips:
            cumulative_perimeter = strip_manager.query_accumulated_perimeter(coord_curr)
            if cumulative_perimeter >= half_perimeter:
                division_point = coord_curr
                break

        if division_point is None:
            # Could not find a half-perimeter crossing => store region
            self.partitions.append((region, obstacles))
            return

        # Divide the region
        divider = ObstacleAwareDivider(strip_manager)
        try:
            (R_left, left_obstacles), (R_right, right_obstacles) = divider.divide_region(division_point)
            logging.info(f"Division successful at {division_point:.4f} along {axis} axis.")
        except ValueError as e:
            logging.error(f"Division failed: {e}")
            self.partitions.append((region, obstacles))
            return

        # Validate subregions
        R_left = validate_geometry(R_left)
        R_right = validate_geometry(R_right)

        if self.advanced_checks:
            if not self._is_subregion_valid(R_left, left_obstacles):
                R_left = None
            if not self._is_subregion_valid(R_right, right_obstacles):
                R_right = None

        # Check if subregions are too narrow or degenerate
        if R_left:
            minx, miny, maxx, maxy = R_left.bounds
            if maxx - minx < self.min_area_threshold or maxy - miny < self.min_area_threshold:
                logging.warning("Left subregion is too narrow or degenerate. Skipping.")
                R_left = None

        if R_right:
            minx, miny, maxx, maxy = R_right.bounds
            if maxx - minx < self.min_area_threshold or maxy - miny < self.min_area_threshold:
                logging.warning("Right subregion is too narrow or degenerate. Skipping.")
                R_right = None

        if not R_left and not R_right:
            logging.warning("Both subregions are invalid. Stopping.")
            self.partitions.append((region, obstacles))
            return

        # Recur on the left and right subregions
        next_axis = 'y' if axis == 'x' else 'x'
        if R_left:
            self.kd_tree_partition(R_left, left_obstacles, depth - 1, next_axis)
        if R_right:
            self.kd_tree_partition(R_right, right_obstacles, depth - 1, next_axis)

    def run(self):
        """Initiate KD-Tree partitioning."""
        self.kd_tree_partition(self.region, self.obstacles, self.max_depth, axis='x')
        logging.info(f"KD-Tree partitioning complete. Total partitions: {len(self.partitions)}")
        return self.partitions

    # ------------------------------------
    #   ADVANCED CHECKS (optional)
    # ------------------------------------
    def _check_coverage_and_stop(self, region, obstacles):
        """
        If coverage≥90% & leftover < DRONE_THRESHOLD => store (stop).
        """
        region_area = region.area
        obs_area = sum(ob.area for ob in obstacles)
        coverage_ratio = obs_area / region_area if region_area > 1e-12 else 1.0
        free_area = region_area - obs_area
        largest_hole_area = self._compute_largest_free_space(region, obstacles)

        if coverage_ratio >= COVERAGE_RATIO_STOP:
            if free_area < DRONE_THRESHOLD and largest_hole_area < DRONE_THRESHOLD:
                return True
        return False

    def _is_subregion_valid(self, region, obstacles):
        """Dimension threshold, coverage check, optional connectivity if advanced_checks=True."""
        if not region or region.is_empty:
            return False

        minx, miny, maxx, maxy = region.bounds
        width = maxx - minx
        height = maxy - miny
        if width < self.min_area_threshold or height < self.min_area_threshold:
            return False

        region_area = region.area
        obs_area_sum = sum(ob.area for ob in obstacles)
        if obs_area_sum >= region_area - 1e-9:
            return False

        if self.check_connectivity:
            largest_hole = self._compute_largest_free_space(region, obstacles)
            if largest_hole < DRONE_THRESHOLD:
                return False

        return True

    def _compute_largest_free_space(self, region, obstacles):
        """Compute largest hole in region after subtracting obstacles."""
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

    def save_partitions(self, datatype, depth, output_dir=f"results/KD_tree", runtime=None, strategies_data=None,
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
        now_str = datetime.now().strftime("%Y%m%d")  # Date only, no time
        csv_filename = os.path.join(output_dir, f"{datatype}_kd_tree_depth_{depth}_{timestamp}.csv")
        tree_filename = os.path.join(output_dir, f"{datatype}_kd_tree_depth_{depth}_{timestamp}_tree.txt")

        # Derive visualization filename based on CSV filename base
        base_filename = os.path.splitext(csv_filename)[0]
        visualization_filename = f"{base_filename}_partition.png"

        # Shared strategies_data file across all depths and percentages
        # Directory-independent summary file
        summary_file = "results/final/AR/kdtree/leftbiased/summary_data.json"
        os.makedirs("results/final/AR/kdtree/leftbiased", exist_ok=True)

        # Initialize summary data if not present
        if not os.path.exists(summary_file):
            summary_data = {}
        else:
            with open(summary_file, "r") as f:
                summary_data = json.load(f)

        # Parse percentage from datatype
        percentage = f"{datatype.split('_')[-1]}%" if "percent" in datatype else "N/A"

        if percentage not in summary_data:
            summary_data[percentage] = {"max_wcrt": [], "std_dev": [], "AR": []}

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

        # Update strategies_data dictionary and save to file
        # Extract WCRT and std_dev from the last row
        summary_data[percentage]["max_wcrt"].append(round(max_wcrt, 2))
        summary_data[percentage]["std_dev"].append(round(std_dev_wcrt, 3))
        summary_data[percentage]["AR"].append(round(sum(aspect_ratios) / len(aspect_ratios), 3) if aspect_ratios else 0)

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

