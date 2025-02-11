import argparse
import logging
import random
import time
import json
from shapely.geometry import shape

from src.data_Loader import DataPreprocessor
from src.kd_tree_naive_decomposition import NaiveKDTreePartitioning
from src.kd_tree_perimeter_decomposition import KDTreePartitioning
from src.preprocessing import RegionWithObstacles

logging.basicConfig(level=logging.INFO)


def main(region, obstacles, max_depth=1, data='synthetic_percent', method='newton', percentage=None):
    """
    Main function to perform recursive partitioning using KD-tree strategies.
    """
    try:
        # Preprocess and validate region and obstacles
        if isinstance(region, str) and isinstance(obstacles, str):
            logging.info("Reading region and obstacles from files...")
            data_loader = DataPreprocessor(region, obstacles)
            region, obstacle_coords_list = data_loader.preprocess()
        else:
            logging.info("Using provided region and obstacle geometries...")
            obstacle_coords_list = [list(ob.exterior.coords) for ob in obstacles]

        logging.info("Initializing RegionWithObstacles...")
        region_with_obstacles = RegionWithObstacles(region, obstacle_coords_list)

        region = region_with_obstacles.region
        obstacles = region_with_obstacles.get_simplified_obstacles()

        # Perform partitioning using KD-tree strategies
        # Half-Perimeter KD-tree
        start_time_kd = time.perf_counter()
        kd_tree = KDTreePartitioning(region, obstacles, max_depth=max_depth, advanced_checks=False)
        kd_tree.run()
        end_time_kd = time.perf_counter()
        runtime_kd = end_time_kd - start_time_kd
        kd_tree.save_partitions(datatype=f"{data}_{percentage}percent" if percentage else data,
                                depth=max_depth, runtime=runtime_kd)

        # Naive KD-tree
        start_time_naive = time.perf_counter()
        naive_kd_tree = NaiveKDTreePartitioning(region, obstacles, max_depth=max_depth, advanced_checks=False)
        naive_kd_tree.run()
        end_time_naive = time.perf_counter()
        runtime_naive = end_time_naive - start_time_naive
        naive_kd_tree.save_partitions(datatype=f"{data}_{percentage}percent" if percentage else data,
                                      depth=max_depth, runtime=runtime_naive)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run recursive partitioning with KD-tree strategies.")
    parser.add_argument("--data_type", type=str, choices=['iowa', 'synthetic', 'synthetic_percent'], required=False,
                        help="Type of data to load.")
    parser.add_argument("--region_file", type=str, help="Path to the region GeoJSON file.")
    parser.add_argument("--obstacles_file", type=str, help="Path to the obstacles GeoJSON file.")

    args = parser.parse_args()

    # Default settings
    data_type = args.data_type or 'synthetic_percent'
    depths = [2, 3, 4, 5, 6]
    # depths = [2]

    method = 'newton'

    if data_type == 'iowa':
        region_file_path = 'resource/dataset/iowa/IOWA-BOUNDARY.geojson'
        obstacles_file_path = 'resource/dataset/iowa/FAA-IOWA.geojson'
        for depth in depths:
            main(region_file_path, obstacles_file_path, depth, data=data_type, method=method)

    elif data_type == 'synthetic':
        synthetic_file_path = "resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json"
        try:
            logging.info(f"Loading synthetic data from {synthetic_file_path}...")
            with open(synthetic_file_path, "r") as f:
                data = json.load(f)

            region = shape(data["region"])
            obstacles = [shape(obstacle) for obstacle in data["obstacles"]]

            for depth in depths:
                main(region, obstacles, depth, data=data_type, method=method)

        except FileNotFoundError:
            logging.error(f"Synthetic data file {synthetic_file_path} not found. Please generate the synthetic data first.")

    elif data_type == 'synthetic_percent':
        # obstacle_percentages = range(0, 31, 5)  # From 0% to 30% in increments of 5 this is for dynamic data generation ranging from 5 to 30
        obstacle_percentages = [5]  # From 0% to 30% in increments of 5 #this is for static 5 % data
        for percentage in obstacle_percentages:
            # file_path = f"synthetic_data_generation/{percentage}_percent_obstacles.json"
            file_path = "resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json"

            try:
                logging.info(f"Loading synthetic_percent data for {percentage}% obstacles from {file_path}...")
                with open(file_path, "r") as f:
                    data = json.load(f)

                region = shape(data["region"])
                obstacles = [shape(obstacle) for obstacle in data["obstacles"]]

                for depth in depths:
                    main(region, obstacles, depth, data=data_type, method=method, percentage=percentage)

            except FileNotFoundError:
                logging.error(f"synthetic_percent data file {file_path} not found. Please generate the synthetic_percent data for {percentage}% obstacles first.")

    else:
        raise ValueError(f"Unsupported data type: {data_type}")
