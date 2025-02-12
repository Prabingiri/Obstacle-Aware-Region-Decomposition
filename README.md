# Obstacle-Aware-Region-Decomposition
MDM-2025

This repository implements a set of algorithms for the hierarchical decomposition of spatial regions with obstacles. This is part of our research into optimal spatial subdivision; the project compares multiple partitioning strategies.

## Overview

The project includes two main pipelines:

- **Sweepline Based Hierarchical Decomposition (main_decomposition.py):**  
  Our primary pipeline incorporates modules for preprocessing, obstacle-aware division, and optimal axis selection to recursively decompose a region into navigable subregions. It is designed for robust performance on both real-world (Iowa) and synthetic datasets.

- **KD‑Tree Partitioning (main_kd_tree.py):**  
  This pipeline implements two classic KD‑Tree–based partitioning strategies (including both naive and perimeter‑based approaches) to serve as a performance baseline for our primary hierarchical decomposition method.

## Project Structure

- **src/**  
  Contains all source code modules. Key modules include:
  - `preprocessing.py` – Implements the `RegionWithObstacles` class for geometry validation and obstacle clipping.
  - `hierarchical_decomposition_algorithm.py` – Contains the `HierarchicalDecomposition` class for recursive decomposition.
  - `obstacle_aware_divider.py` – Provides methods for dividing regions using numerical root-finding and event-based strategies.
  - `optimal_axis_selection.py` – Implements the `OptimalAxisSelection` class to choose the optimal axis for partitioning.
  - `strip_perimeter.py` – Contains the `Strip` class for computing perimeters and related metrics.
  - `kd_tree_naive_decomposition.py` and `kd_tree_perimeter_decomposition.py` – Implement alternative KD‑Tree partitioning strategies.
  - `save_partitions.py` – Provides functions to save partitioning results (CSV files, text trees, and visualizations).

- **resource/**

  It contains the resources, datasets, and functions to generate synthetic datasets. Check out its readme file.

- **tests/**  
  Contains unit tests for each module to ensure correctness.

- **main_decomposition.py**  
  The main entry point for running the hierarchical decomposition pipeline.

- **main_kd_tree.py**  
  The main entry point for running the KD‑Tree partitioning pipeline for comparison.

- **resource/** and **synthetic_data_generation/**  
  Contain sample datasets (GeoJSON files and synthetic data) used for testing and demonstration.

## Dependencies

- **Python 3.7+**
- **Shapely** (for geometric operations)
- **rtree** (for spatial indexing in some modules)
- **Matplotlib** (for visualization)
- Additional standard libraries: `json`, `csv`, `argparse`, `logging`, etc.

It is recommended to create a virtual environment (using `venv` or `conda`) and install all required dependencies via the provided `requirements.txt` file.

## Running the Project

### Hierarchical Decomposition Pipeline

To run the primary hierarchical decomposition pipeline, use:

```bash
python main_decomposition.py --data_type <data_type> --region_file <path_to_region_file> --obstacles_file <path_to_obstacles_file> --output_dir <output_directory>
```
### Note:

For the Iowa dataset, region and obstacle inputs are provided directly as geometry objects.
For synthetic data, the JSON files should contain a top‑level "region" key (for the region geometry) and an appropriate structure for obstacles.

##### Run synthetic dataset
```bash
python main_decomposition.py --data_type synthetic_percent --region_file resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json --obstacles_file resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json --output_dir results
```
##### Run KD‑Tree Partitioning Pipeline
To run the KD‑Tree partitioning pipeline for comparison, execute
  ```bash
  python main_kd_tree.py --data_type <data_type> --region_file <path_to_region_file> --obstacles_file <path_to_obstacles_file>
eg: python main_kd_tree.py --data_type synthetic_percent --region_file resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json --obstacles_file resource/dataset/synthetic_data_generated/100x100/synthetic_data_5percent_50maxobs_1var.json
```
