import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional

class DepthMetricsComparator:
    """
    A class to traverse each 'depth_*' directory under 'base_path',
    parse CSV files, and extract the last-row user metrics:
      - WCRT
      - aspect_ratio
      - min-wcrt
      - max-wcrt
      - variance
      - standard_deviation
      - range
    We then can save these data to a CSV, create line plots,
    and highlight which metrics stand out for each measure.
    """

    def __init__(self, base_path: str = "synthetic"):
        """
        :param base_path: The root directory which contains subdirectories
                          named 'depth_2', 'depth_3', etc.
        """
        self.base_path = base_path

        # Data structure to hold results in the form:
        # {
        #   depth_value: {
        #       "BWCRT":  { "WCRT": ..., "aspect_ratio": ..., ... },
        #       "IRTE":   { "WCRT": ..., "aspect_ratio": ..., ... },
        #       "MAD":    { ... },
        #       "Variance": { ... },
        #       ...
        #   },
        #   ...
        # }
        self.results: Dict[int, Dict[str, Dict[str, float]]] = {}

        # Columns we care about (from each CSV's last row)
        self.columns_of_interest = [
            "WCRT",
            "aspect_ratio",
            "min-wcrt",
            "max-wcrt",
            "variance",
            "standard_deviation",
            "range"
        ]

    def collect_metrics(self) -> None:

        if not os.path.isdir(self.base_path):
            raise NotADirectoryError(f"Base path '{self.base_path}' is not a directory.")

        # Iterate over items in base_path
        for item in os.listdir(self.base_path):
            depth_dir_path = os.path.join(self.base_path, item)
            if os.path.isdir(depth_dir_path) and item.startswith("depth_"):
                # Extract the depth number from folder name "depth_XYZ"
                try:
                    depth_value = int(item.split("_")[1])
                except (IndexError, ValueError):
                    continue

                # Initialize if not present
                if depth_value not in self.results:
                    self.results[depth_value] = {}

                # Now look through each file in the depth_X folder
                for filename in os.listdir(depth_dir_path):
                    if not filename.endswith(".csv"):
                        # Ignore non-CSV (like .tree.txt)
                        continue

                    full_csv_path = os.path.join(depth_dir_path, filename)

                    # Example filename: "synthetic_newton_BWCRT_depth2_20250103_232619.csv"
                    # We'll assume the user_metric (e.g. 'BWCRT') is at index 2
                    parts = filename.split("_")
                    if len(parts) < 4:
                        # Filenames that don't match the expected pattern are skipped
                        continue

                    metric_name = parts[2]  # e.g. "BWCRT", "IRTE", "MAD", "Variance", etc.

                    # Read the CSV
                    try:
                        df = pd.read_csv(full_csv_path)
                    except Exception as e:
                        print(f"Warning: Could not read {full_csv_path}: {e}")
                        continue

                    # We want the last row (overall aggregated data)
                    if df.empty:
                        continue

                    last_row = df.iloc[-1]

                    # Extract columns of interest
                    extracted = {}
                    for col in self.columns_of_interest:
                        extracted[col] = last_row[col] if col in last_row else None

                    # Insert data into results
                    self.results[depth_value][metric_name] = extracted

    def to_dataframe(self) -> pd.DataFrame:
        """
        Converts the self.results structure into a single pandas DataFrame
        with columns: [depth, metric_name, WCRT, aspect_ratio, min-wcrt, max-wcrt,
        variance, standard_deviation, range].
        """
        rows = []
        for depth_value in sorted(self.results.keys()):
            for metric_name, col_values in self.results[depth_value].items():
                row_data = {
                    "depth": depth_value,
                    "metric_name": metric_name,
                }
                for col in self.columns_of_interest:
                    row_data[col] = col_values.get(col, None)
                rows.append(row_data)

        df = pd.DataFrame(rows)
        return df

    def save_to_csv(self, out_csv: str = "all_depth_metrics.csv") -> None:
        """
        Saves the aggregated DataFrame to a CSV file.
        """
        df = self.to_dataframe()
        df.to_csv(out_csv, index=False)
        print(f"Saved merged metrics to: {out_csv}")

    def plot_all_columns(self, output_prefix: str = "plot_") -> None:
        """
        For each of the columns_of_interest, this method:
          1) Pivots the data so that X-axis = depth, lines = different metrics.
          2) Plots line charts.
          3) Saves each plot as a PNG file named e.g. "plot_WCRT.png".
        """
        df = self.to_dataframe()

        # For each column, pivot the table so:
        #    index='depth'
        #    columns='metric_name'
        #    values=<that column>
        # Then plot a line chart
        for col in self.columns_of_interest:
            # If the column is all None, skip
            if df[col].isnull().all():
                print(f"No data for column '{col}' - skipping plot.")
                continue

            pivot_df = df.pivot(index="depth", columns="metric_name", values=col)
            # Create a line plot
            ax = pivot_df.plot(kind="line", marker="o", figsize=(8, 5), title=f"{col} vs. Depth")
            ax.set_xlabel("Depth")
            ax.set_ylabel(col)
            plt.legend(title="Metric")

            # Save figure
            out_png = f"{output_prefix}{col}.png"
            plt.savefig(out_png, bbox_inches="tight")
            plt.close()
            print(f"Saved plot for '{col}' to: {out_png}")

    def compare_all(self) -> None:
        """
        Prints a summary of all results to the console, grouped by depth.
        """
        for depth_value in sorted(self.results.keys()):
            print(f"Depth {depth_value}:")
            for metric_name, data_dict in self.results[depth_value].items():
                print(f"  Metric: {metric_name}")
                for col_name, val in data_dict.items():
                    print(f"    {col_name} = {val}")
            print()

    def highlight_standouts(
            self,
            measure: str = "WCRT",
            mode: str = "min"
    ) -> pd.DataFrame:
        """
        For each depth, find which metric(s) is the best or the worst for a given measure.

        :param measure: The column name to compare (e.g., 'WCRT', 'aspect_ratio', etc.).
        :param mode: 'min' to pick the metric(s) with the smallest values,
                     'max' to pick the largest values.
        :return: A DataFrame of standouts, one row per 'winning' metric at each depth.

        Example:
            If measure='WCRT' and mode='min', we find the metric(s) with
            the *lowest* WCRT at each depth.
        """
        df = self.to_dataframe()

        # If the requested measure isn't in the DataFrame, just return empty
        if measure not in df.columns:
            print(f"Measure '{measure}' not found in data. Returning empty.")
            return pd.DataFrame()

        # Group by depth
        group_by_depth = df.groupby("depth")

        standout_rows = []
        for depth_val, group in group_by_depth:
            # Filter out rows that don't have a valid measure
            group = group.dropna(subset=[measure])
            if group.empty:
                continue

            # Find min or max value
            if mode == "min":
                standout_value = group[measure].min()
            else:
                standout_value = group[measure].max()

            # Find which row(s) has that standout value
            best_rows = group[group[measure] == standout_value]

            # Add them to our list
            for _, row in best_rows.iterrows():
                standout_rows.append(row.to_dict())

        # Convert to DataFrame for convenience
        standout_df = pd.DataFrame(standout_rows)
        # Sort by depth ascending, for readability
        standout_df = standout_df.sort_values(by="depth")
        return standout_df


def main():
    comparator = DepthMetricsComparator(base_path="results/iowa")

    # 1. Collect metrics from the CSV files
    comparator.collect_metrics()

    # 2. Print them all (optional, for checking)
    comparator.compare_all()

    # 3. Save the aggregated data to a single CSV
    comparator.save_to_csv("all_depth_metrics.csv")

    # 4. Create line plots for each column across depths
    comparator.plot_all_columns(output_prefix="plot_")

    # 5. Show which metric stands out for a given measure
    #    Example 1: Which metric yields the lowest WCRT at each depth?
    standout_df_min_wcrt = comparator.highlight_standouts(measure="WCRT", mode="min")
    print("\nMetrics with the *lowest* WCRT at each depth:")
    print(standout_df_min_wcrt)

    #    Example 2: Which metric yields the highest aspect ratio at each depth?
    standout_df_max_ar = comparator.highlight_standouts(measure="aspect_ratio", mode="max")
    print("\nMetrics with the *highest* aspect_ratio at each depth:")
    print(standout_df_max_ar)

    # 6. (New) Example 3: Which metric yields the *lowest variance* at each depth?
    standout_df_min_variance = comparator.highlight_standouts(measure="variance", mode="min")
    print("\nMetrics with the *lowest* variance at each depth:")
    print(standout_df_min_variance)

    print("\nAll done.")


if __name__ == "__main__":
    main()
