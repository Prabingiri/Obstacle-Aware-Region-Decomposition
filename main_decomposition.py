import argparse
import logging
import json
import time
from shapely.geometry import shape
from src.hierarchical_decomposition_algorithm import HierarchicalDecomposition
from src.preprocessing import RegionWithObstacles
from src.save_partitions import save_final_results

logging.basicConfig(level=logging.INFO)


def main(region, obstacles, max_depth, numerical_method, dataset=None, percentage=None, **kwargs):
    """
    Main function to run hierarchical decomposition.
    """
    try:
        if isinstance(region, str) and isinstance(obstacles, str):
            logging.info("Reading region + obstacles from files...")
            with open(region, "r") as f:
                region_data = json.load(f)
            region_geom = shape(region_data["region"])

            with open(obstacles, "r") as f:
                obstacles_data = json.load(f)
            obstacles_list = [shape(ob) for ob in obstacles_data]
        else:
            region_geom = region
            obstacles_list = obstacles

        region_with_obstacles = RegionWithObstacles(region_geom, obstacles_list)
        region_final = region_with_obstacles.region
        obstacles_final = region_with_obstacles.get_simplified_obstacles()

        chosen_metric = kwargs.get("metric", "NWCRT")

        start_time = time.perf_counter()
        decomposition = HierarchicalDecomposition(
            region_final,
            obstacles_final,
            max_depth=max_depth,
            metrics=chosen_metric,
            numerical_method=numerical_method,
            min_dimension_threshold=1e-3,
            check_connectivity=False,
            allow_fallback_axis=True,
            mode="track_back"
        )
        partitions = decomposition.run()
        end_time = time.perf_counter()
        runtime = end_time - start_time

        dataset_tag = f"{dataset}_{percentage}percent" if percentage is not None else dataset
        data = dataset
        save_final_results(
            partitions=partitions,
            datatype=dataset_tag,
            numerical_method=numerical_method,
            user_metric=chosen_metric,
            depth=max_depth,
            # output_dir=f"results/{dataset_tag}/depth_{max_depth}",
            output_dir=f"results/obstacle_aware/{dataset_tag}/depth_{max_depth}",
            runtime=runtime
        )
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hierarchical partitioning pipeline.")
    parser.add_argument("--data_type", type=str, choices=['iowa', 'synthetic', 'synthetic_percent'], required=False,
                        help="Which dataset to load.")
    parser.add_argument("--region_file", type=str, help="Path to region GeoJSON.")
    parser.add_argument("--obstacles_file", type=str, help="Path to obstacles GeoJSON.")
    parser.add_argument("--numerical_method", type=str, choices=['brent', 'newton'], default="newton",
                        help="Root-finding method.")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save the output files.")

    args = parser.parse_args()

    data_type = args.data_type or 'synthetic_percent'
    numerical_method = args.numerical_method
    output_dir = args.output_dir

    depths = [2, 3, 4, 5, 6]
    metrics = ["NWCRT"]
    method = 'newton'

    if data_type == 'iowa':
        # provide the file path needed
        region_file = 'resource/dataset/iowa/IOWA-BOUNDARY.geojson'
        obs_file = 'resource/dataset/iowa/FAA-IOWA.geojson'
        for metric in metrics:
            for depth in depths:
                logging.info(f"Running for metric={metric}, depth={depth}")
                main(region_file, obs_file, depth, numerical_method,
                     dataset=data_type, metric=metric, output_dir=output_dir)

    elif data_type == 'synthetic':
        # provide the file path needed
        synthetic_file_path = "resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json"
        try:
            logging.info(f"Loading synthetic data from {synthetic_file_path}...")
            with open(synthetic_file_path, "r") as f:
                data = json.load(f)

            region_geom = shape(data["region"])
            obstacles_list = [shape(ob) for ob in data["obstacles"]]

            for metric in metrics:
                for depth in depths:
                    logging.info(f"Running for metric={metric}, depth={depth}")
                    main(region_geom, obstacles_list, depth, numerical_method,
                         dataset=data_type, metric=metric, output_dir=output_dir)
        except FileNotFoundError:
            logging.error(f"Synthetic data file {synthetic_file_path} not found.")

    elif data_type == 'synthetic_percent':
        # obstacle_percentages = range(0, 31, 5)  # From 0% to 30% in increments of 5
        obstacle_percentages = [5]  # From 0% to 30% in increments of 5
        for percentage in obstacle_percentages:
            #provide the file path needed
            file_path =  "resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json"
            # file_path = f"synthetic_data_generation/{percentage}_percent_obstacles.json"

            try:
                logging.info(f"Loading synthetic_percent data for {percentage}% obstacles from {file_path}...")
                with open(file_path, "r") as f:
                    data = json.load(f)

                region = shape(data["region"])
                obstacles = [shape(obstacle) for obstacle in data["obstacles"]]

                for depth in depths:
                    main(region, obstacles, depth, numerical_method=method, data=data_type, dataset=data_type, metric=metrics, percentage=percentage)

            except FileNotFoundError:
                logging.error(f"synthetic_percent data file {file_path} not found. Please generate the synthetic_percent data for {percentage}% obstacles first.")


    else:
        logging.error(f"Unsupported data type: {data_type}")
