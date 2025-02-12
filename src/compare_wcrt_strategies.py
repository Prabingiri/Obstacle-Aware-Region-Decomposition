"""
File: compare_wcrt_strategies.py

Description:
    A module to compare naive KD tree, KD-tree (perimeter) decomposition vs. Obstacle-Aware decomposition
    across multiple depths, plotting the following metrics:

      1) Avg WCRT + Max WCRT (together)
      2) Min WCRT + Mean-Square Deviation (together)
      3) Aspect Ratio (alone)
      4) execution time

    We produce two styles of plots for each grouping:
      - bar plots
      - line plots
    plotter.plot_aspect_ratio_line(log_scale=False)
"""

import matplotlib.pyplot as plt
import numpy as np


class WCRTComparisonPlotter:
    def __init__(
        self,
        depths,
        # --- Naive KD metrics ---
        naive_avg_wcrt=None, naive_min_wcrt=None, naive_max_wcrt=None, naive_msdev=None,
        naive_aspect_ratio=None,
        naive_exec_time=None,

        # --- KD (Perimeter) metrics (original "kd_" approach) ---
        kd_avg_wcrt=None, kd_min_wcrt=None, kd_max_wcrt=None, kd_msdev=None,
        kd_aspect_ratio=None,
        kd_exec_time=None,

            # --- Obstacle-Aware metrics (original "obs_" approach) ---
        obs_avg_wcrt=None, obs_min_wcrt=None, obs_max_wcrt=None, obs_msdev=None,
        obs_aspect_ratio=None,
        obs_exec_time = None
    ):
        """
        Minimal changes: we simply store all arrays as before,
        but we'll produce 3 separate images for each approach (bar & line).
        """
        self.depths = depths

        # Naive KD
        self.naive_avg_wcrt = naive_avg_wcrt
        self.naive_min_wcrt = naive_min_wcrt
        self.naive_max_wcrt = naive_max_wcrt
        self.naive_msdev    = naive_msdev
        self.naive_aspect   = naive_aspect_ratio
        self.naive_exec     = naive_exec_time

        # KD-tree metrics
        self.kd_avg_wcrt = kd_avg_wcrt
        self.kd_min_wcrt = kd_min_wcrt
        self.kd_max_wcrt = kd_max_wcrt
        self.kd_msdev    = kd_msdev
        self.kd_aspect   = kd_aspect_ratio  # might be None
        self.kd_exec = kd_exec_time

        # Obstacle-Aware metrics
        self.obs_avg_wcrt = obs_avg_wcrt
        self.obs_min_wcrt = obs_min_wcrt
        self.obs_max_wcrt = obs_max_wcrt
        self.obs_msdev    = obs_msdev
        self.obs_aspect   = obs_aspect_ratio  # might be None
        self.obs_exec = obs_exec_time

    # --------------------------------------------------------------------
    #  1) AVG + MAX WCRT (Bar)
    # --------------------------------------------------------------------
    def plot_avg_and_max_wcrt_bar(self, log_scale=False):
        """
        Creates a single figure with 2 subplots (both bar):
          Subplot 1: KD-tree vs. Obstacle-Aware for Avg WCRT
          Subplot 2: KD-tree vs. Obstacle-Aware for Max WCRT
        """
        x = np.arange(len(self.depths))
        width = 0.35

        fig, (ax_avg, ax_max) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # 1) Bar: Avg WCRT
        ax_avg.bar(x - width/2, self.kd_avg_wcrt, width, label='KD-tree')
        ax_avg.bar(x + width/2, self.obs_avg_wcrt, width, label='Obstacle-Aware')
        ax_avg.set_xticks(x)
        ax_avg.set_xticklabels([str(d) for d in self.depths])
        ax_avg.set_title("Avg WCRT", fontsize=14)
        ax_avg.set_xlabel("Depth", fontsize=12)
        ax_avg.set_ylabel("Avg WCRT", fontsize=12)
        ax_avg.legend(fontsize=11)
        if log_scale:
            ax_avg.set_yscale('log')

        # 2) Bar: Max WCRT
        ax_max.bar(x - width/2, self.kd_max_wcrt, width, label='KD-tree')
        ax_max.bar(x + width/2, self.obs_max_wcrt, width, label='Obstacle-Aware')
        ax_max.set_xticks(x)
        ax_max.set_xticklabels([str(d) for d in self.depths])
        ax_max.set_title("Max WCRT (Bar)", fontsize=14)
        ax_max.set_xlabel("Depth", fontsize=12)
        ax_max.set_ylabel("Max WCRT", fontsize=12)
        ax_max.legend(fontsize=11)
        if log_scale:
            ax_max.set_yscale('log')

        fig.suptitle("Avg & Max WCRT Comparison (Bar)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    #  2) AVG + MAX WCRT (Line)
    # --------------------------------------------------------------------
    def plot_avg_and_max_wcrt_line(self, log_scale=False):
        """
        Similar approach: 1 figure w/ 2 subplots for line plots:
          Subplot 1 => Avg WCRT
          Subplot 2 => Max WCRT
        """
        x = self.depths
        fig, (ax_avg, ax_max) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # 1) Avg WCRT line
        ax_avg.plot(x, self.kd_avg_wcrt, 'o--', label='KD-tree')
        ax_avg.plot(x, self.obs_avg_wcrt, 's--', label='Obstacle-Aware')
        ax_avg.set_title("Avg WCRT (Line)", fontsize=14)
        ax_avg.set_xlabel("Depth", fontsize=12)
        ax_avg.set_ylabel("Avg WCRT", fontsize=12)
        ax_avg.legend(fontsize=11)
        if log_scale:
            ax_avg.set_yscale('log')

        # 2) Max WCRT line
        ax_max.plot(x, self.kd_max_wcrt, 'o--', color='tab:orange', label='KD-tree')
        ax_max.plot(x, self.obs_max_wcrt, 's--', color='tab:green', label='Obstacle-Aware')
        ax_max.set_title("Max WCRT (Line)", fontsize=14)
        ax_max.set_xlabel("Depth", fontsize=12)
        ax_max.set_ylabel("Max WCRT", fontsize=12)
        ax_max.legend(fontsize=11)
        if log_scale:
            ax_max.set_yscale('log')

        fig.suptitle("Avg & Max WCRT Comparison (Line)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    #  3) MIN WCRT + MS Deviation (Bar)
    # --------------------------------------------------------------------
    def plot_min_and_msdev_bar(self, log_scale=False):
        """
        Single figure w/ 2 subplots (bar) for:
          Subplot 1 => Min WCRT
          Subplot 2 => Mean-Square Deviation
        """
        x = np.arange(len(self.depths))
        width = 0.35

        fig, (ax_min, ax_msdev) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # 1) Min WCRT
        ax_min.bar(x - width/2, self.kd_min_wcrt, width, label='KD-tree')
        ax_min.bar(x + width/2, self.obs_min_wcrt, width, label='Obstacle-Aware')
        ax_min.set_xticks(x)
        ax_min.set_xticklabels([str(d) for d in self.depths])
        ax_min.set_title("Min WCRT (Bar)", fontsize=14)
        ax_min.set_xlabel("Depth", fontsize=12)
        ax_min.set_ylabel("Min WCRT", fontsize=12)
        ax_min.legend(fontsize=11)
        if log_scale:
            ax_min.set_yscale('log')

        # 2) Mean-Square Deviation
        ax_msdev.bar(x - width/2, self.kd_msdev, width, label='KD-tree')
        ax_msdev.bar(x + width/2, self.obs_msdev, width, label='Obstacle-Aware')
        ax_msdev.set_xticks(x)
        ax_msdev.set_xticklabels([str(d) for d in self.depths])
        ax_msdev.set_title("Mean Square Deviation", fontsize=14)
        ax_msdev.set_xlabel("Depth", fontsize=12)
        ax_msdev.set_ylabel("Mean-Square Deviation", fontsize=12)
        ax_msdev.legend(fontsize=11)
        if log_scale:
            ax_msdev.set_yscale('log')

        fig.suptitle("Min WCRT & MS Deviation (Bar)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    #  4) MIN WCRT + MS Deviation (Line)
    # --------------------------------------------------------------------
    def plot_min_and_msdev_line(self, log_scale=False):
        """
        Single figure w/ 2 subplots (line) for:
          Subplot 1 => Min WCRT
          Subplot 2 => Mean-Square Deviation
        """
        x = self.depths
        fig, (ax_min, ax_msdev) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

        # 1) Min WCRT
        ax_min.plot(x, self.kd_min_wcrt, 'o--', label='KD-tree')
        ax_min.plot(x, self.obs_min_wcrt, 's--', label='Obstacle-Aware')
        ax_min.set_title("Min WCRT (Line)", fontsize=14)
        ax_min.set_xlabel("Depth", fontsize=12)
        ax_min.set_ylabel("Min WCRT", fontsize=12)
        ax_min.legend(fontsize=11)
        if log_scale:
            ax_min.set_yscale('log')

        # 2) MS Dev
        ax_msdev.plot(x, self.kd_msdev, 'o--', label='KD-tree')
        ax_msdev.plot(x, self.obs_msdev, 's--', label='Obstacle-Aware')
        ax_msdev.set_title("MS Dev (Line)", fontsize=14)
        ax_msdev.set_xlabel("Depth", fontsize=12)
        ax_msdev.set_ylabel("Mean-Square Deviation", fontsize=12)
        ax_msdev.legend(fontsize=11)
        if log_scale:
            ax_msdev.set_yscale('log')

        # fig.suptitle("Min WCRT & MS Deviation (Line)", fontsize=16)
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    #  5) ASPECT RATIO (Bar)
    # --------------------------------------------------------------------
    def plot_aspect_ratio_bar(self, log_scale=False):
        """
        If kd_aspect_ratio or obs_aspect_ratio is None, skip.
        Otherwise, single figure with bar chart for KD vs. Obs.
        """
        if self.kd_aspect is None or self.obs_aspect is None:
            print("Aspect ratio data not provided => skipping bar.")
            return

        x = np.arange(len(self.depths))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - width/2, self.kd_aspect,  width, color='tab:orange', label='Perimeter-Based ')
        ax.bar(x + width/2, self.obs_aspect, width, color='tab:green', label='Obstacle-Aware')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in self.depths])
        # ax.set_title("Aspect Ratio (Bar)", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Aspect Ratio", fontsize=12)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Aspect Ratio = 0.5')
        ax.legend(fontsize=11)
        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------------------
    #  6) ASPECT RATIO (Line)
    # --------------------------------------------------------------------
    def plot_aspect_ratio_line(self, log_scale=False):
        """
        If kd_aspect_ratio or obs_aspect_ratio is None, skip.
        Otherwise, single figure with line chart for KD vs. Obs.
        """
        if self.kd_aspect is None or self.obs_aspect is None:
            print("Aspect ratio data not provided => skipping line.")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.plot(self.depths, self.kd_aspect, 'o--', label='Perimeter-Based KD')
        ax.plot(self.depths, self.obs_aspect, 's--', label='Obstacle-Aware')
        ax.set_title("Aspect Ratio (Line)", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Aspect Ratio", fontsize=12)
        ax.legend(fontsize=11)
        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()



    def plot_avg_and_max_wcrt_bar_triple(self, log_scale=False, unify_scale=True):
        """
        Bar plots for THREE approaches side by side:
          1) Naive
          2) KD (Perimeter)
          3) Obstacle-Aware
        Subplot 1 => Average WCRT
        Subplot 2 => Maximum WCRT

        If 'unify_scale=True' (and 'log_scale=False'), we unify the y-limits
        so both subplots share the same vertical range.
        """
        # 1) Check data
        if (self.naive_avg_wcrt is None) or (self.kd_avg_wcrt is None) or (self.obs_avg_wcrt is None):
            print("Missing triple data for Avg WCRT. Skipping.")
            return
        if (self.naive_max_wcrt is None) or (self.kd_max_wcrt is None) or (self.obs_max_wcrt is None):
            print("Missing triple data for Max WCRT. Skipping.")
            return

        # 2) Create subplots
        x = np.arange(len(self.depths))
        width = 0.2
        fig, (ax_max, ax_avg) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        # 3) Plot Average WCRT
        ax_avg.bar(x - width, self.naive_avg_wcrt, width, color='tab:blue', label='Naive KD')
        ax_avg.bar(x, self.kd_avg_wcrt, width, color='tab:orange', label='Perimeter-Based KD')
        ax_avg.bar(x + width, self.obs_avg_wcrt, width, color='tab:green', label='Obstacle-Aware')
        ax_avg.set_xticks(x)
        ax_avg.set_xticklabels([str(d) for d in self.depths])
        # ax_avg.set_title("Average WCRT", fontsize=14)
        ax_avg.set_xlabel("Depth", fontsize=12)
        ax_avg.set_ylabel("Avg WCRT", fontsize=12)
        ax_avg.legend(fontsize=11)

        # 4) Plot Maximum WCRT
        ax_max.bar(x - width, self.naive_max_wcrt, width, color='tab:blue', label='Naive KD')
        ax_max.bar(x, self.kd_max_wcrt, width, color='tab:orange', label='Perimeter-Based KD')
        ax_max.bar(x + width, self.obs_max_wcrt, width, color='tab:green', label='Obstacle-Aware')
        ax_max.set_xticks(x)
        ax_max.set_xticklabels([str(d) for d in self.depths])
        # ax_max.set_title("Maximum WCRT", fontsize=14)
        ax_max.set_xlabel("Depth", fontsize=12)
        ax_max.set_ylabel("Max WCRT", fontsize=12)
        ax_max.legend(fontsize=11)

        # 5) Handle log-scale vs. unified linear scale
        if log_scale:
            # If user wants a log-scale, we typically let each subplot auto-scale.
            ax_avg.set_yscale('log')
            ax_max.set_yscale('log')
        elif unify_scale:
            # If we want to unify the linear y-scale => find a global maximum
            global_max = max(
                max(self.naive_avg_wcrt), max(self.naive_max_wcrt),
                max(self.kd_avg_wcrt), max(self.kd_max_wcrt),
                max(self.obs_avg_wcrt), max(self.obs_max_wcrt)
            )
            # e.g., set a bit of margin
            upper_lim = global_max * 1.1
            ax_avg.set_ylim(0, upper_lim)
            ax_max.set_ylim(0, upper_lim)

        # fig.suptitle("Avg & Max WCRT: Naive vs. KD vs. Obstacle (Bar)", fontsize=15)
        plt.tight_layout()
        plt.show()

    def plot_avg_and_max_wcrt_line_triple(self, log_scale=False, unify_scale=True):
        """
        Line plots for THREE approaches:
          Subplot 1 => Average WCRT
          Subplot 2 => Maximum WCRT
        """
        if (self.naive_avg_wcrt is None) or (self.kd_avg_wcrt is None) or (self.obs_avg_wcrt is None):
            print("Missing triple data for Avg WCRT. Skipping.")
            return
        if (self.naive_max_wcrt is None) or (self.kd_max_wcrt is None) or (self.obs_max_wcrt is None):
            print("Missing triple data for Max WCRT. Skipping.")
            return

        x = self.depths
        fig, (ax_max, ax_avg) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        # 1) Avg WCRT line
        ax_avg.plot(x, self.naive_avg_wcrt, 'o--', color='tab:blue', label='Naive KD')
        ax_avg.plot(x, self.kd_avg_wcrt, 's--', color='tab:orange', label='Perimeter-Based KD')
        ax_avg.plot(x, self.obs_avg_wcrt, '^--', color='tab:green', label='Obstacle-Aware')
        # ax_avg.set_title("Average WCRT", fontsize=14)
        ax_avg.set_xlabel("Depth", fontsize=12)
        ax_avg.set_ylabel("Avg WCRT", fontsize=12)
        ax_avg.legend(fontsize=11)
        if log_scale:
            ax_avg.set_yscale('log')

        # 2) Max WCRT line
        ax_max.plot(x, self.naive_max_wcrt, 'o--', color='tab:blue', label='Naive KD')
        ax_max.plot(x, self.kd_max_wcrt, 's--', color='tab:orange',  label='Perimeter KD')
        ax_max.plot(x, self.obs_max_wcrt, '^--', color='tab:green', label='Obstacle-Aware')
        # ax_max.set_title("Maximum WCRT", fontsize=14)
        ax_max.set_xlabel("Depth", fontsize=12)
        ax_max.set_ylabel("Max WCRT", fontsize=12)
        ax_max.legend(fontsize=11)
        if log_scale:
            ax_max.set_yscale('log')

        elif unify_scale:
            # If we want one linear y-scale, find a single global max across both sets
            global_max = max(
                max(self.naive_max_wcrt), max(self.kd_max_wcrt), max(self.obs_max_wcrt),
                max(self.naive_avg_wcrt), max(self.kd_avg_wcrt), max(self.obs_avg_wcrt)
            )
            upper_lim = global_max * 1.1
            ax_max.set_ylim(0, upper_lim)
            ax_avg.set_ylim(0, upper_lim)

        # fig.suptitle("Avg & Max WCRT: Naive vs. KD vs. Obstacle ", fontsize=15)
        plt.tight_layout()
        plt.show()

    def plot_min_and_msdev_bar_triple(self, log_scale=False):
        """
        Bar plots for THREE approaches:
          Subplot 1 => Min WCRT
          Subplot 2 => Mean-Square Dev
        """
        if (self.naive_min_wcrt is None) or (self.kd_min_wcrt is None) or (self.obs_min_wcrt is None):
            print("Missing triple data for Min WCRT. Skipping.")
            return
        if (self.naive_msdev is None) or (self.kd_msdev is None) or (self.obs_msdev is None):
            print("Missing triple data for MS Dev. Skipping.")
            return

        x = np.arange(len(self.depths))
        width = 0.2

        fig, (ax_min, ax_msdev) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        # Min WCRT
        ax_min.bar(x - width, self.naive_min_wcrt, width, color='tab:blue', label='Naive KD')
        ax_min.bar(x, self.kd_min_wcrt, width, color='tab:orange', label='Perimeter-Based KD')
        ax_min.bar(x + width, self.obs_min_wcrt, width, color='tab:green', label='Obstacle-Aware')
        ax_min.set_xticks(x)
        ax_min.set_xticklabels([str(d) for d in self.depths])
        # ax_min.set_title("Minimum WCRT", fontsize=14)
        ax_min.set_xlabel("Depth", fontsize=12)
        ax_min.set_ylabel("Min WCRT", fontsize=12)
        ax_min.legend(fontsize=11)
        if log_scale:
            ax_min.set_yscale('log')

        # MS Dev
        ax_msdev.bar(x - width, self.naive_msdev, width, color='tab:blue', label='Naive KD')
        ax_msdev.bar(x, self.kd_msdev, width, color='tab:orange', label='Perimeter-Based KD')
        ax_msdev.bar(x + width, self.obs_msdev, width, color='tab:green', label='Obstacle-Aware')
        ax_msdev.set_xticks(x)
        ax_msdev.set_xticklabels([str(d) for d in self.depths])
        # ax_msdev.set_title("Mean-Square Deviation", fontsize=14)
        ax_msdev.set_xlabel("Depth", fontsize=12)
        ax_msdev.set_ylabel("MS Deviation", fontsize=12)
        ms_all = self.naive_msdev + self.kd_msdev + self.obs_msdev
        mean_ms = sum(ms_all) / len(ms_all)
        ax_msdev.axhline(y=mean_ms, color='purple', linestyle='--', label=f'Mean MSDev={mean_ms:.2f}')

        ax_msdev.legend(fontsize=11)
        if log_scale:
            ax_msdev.set_yscale('log')

        # fig.suptitle("Min WCRT & MS Dev: Naive vs. KD vs. Obstacle (Bar)", fontsize=15)
        plt.tight_layout()
        plt.show()

    def plot_min_and_msdev_line_triple(self, log_scale=False):
        """
        Line plots for THREE approaches:
          Subplot 1 => Min WCRT
          Subplot 2 => Mean-Square Deviation
        """
        if (self.naive_min_wcrt is None) or (self.kd_min_wcrt is None) or (self.obs_min_wcrt is None):
            print("Missing triple data for Min WCRT. Skipping.")
            return
        if (self.naive_msdev is None) or (self.kd_msdev is None) or (self.obs_msdev is None):
            print("Missing triple data for MS Dev. Skipping.")
            return

        x = self.depths
        fig, (ax_min, ax_msdev) = plt.subplots(nrows=1, ncols=2, figsize=(14, 5))

        # Min WCRT line
        ax_min.plot(x, self.naive_min_wcrt, 'o--', color='tab:blue',label='Naive KD')
        ax_min.plot(x, self.kd_min_wcrt, 's--', color='tab:orange', label='Perimeter KD')
        ax_min.plot(x, self.obs_min_wcrt, '^--', color='tab:green', label='Obstacle-Aware')
        # ax_min.set_title("Minimum WCRT", fontsize=14)
        ax_min.set_xlabel("Depth", fontsize=12)
        ax_min.set_ylabel("Min WCRT", fontsize=12)
        ax_min.legend(fontsize=11)
        if log_scale:
            ax_min.set_yscale('log')

        # MS Dev line
        ax_msdev.plot(x, self.naive_msdev, 'o--', color='tab:blue', label='Naive KD')
        ax_msdev.plot(x, self.kd_msdev, 's--', color='tab:orange',  label='Perimeter KD')
        ax_msdev.plot(x, self.obs_msdev, '^--', color='tab:green', label='Obstacle-Aware')
        # ax_msdev.set_title("Mean-Square Deviation", fontsize=14)
        ax_msdev.set_xlabel("Depth", fontsize=12)
        ax_msdev.set_ylabel("MS Deviation", fontsize=12)
        ms_all = self.naive_msdev + self.kd_msdev + self.obs_msdev
        mean_ms = sum(ms_all) / len(ms_all)
        ax_msdev.axhline(y=mean_ms, color='purple', linestyle='--', label=f'Mean MSDev={mean_ms:.2f}')

        ax_msdev.legend(fontsize=11)
        if log_scale:
            ax_msdev.set_yscale('log')

        # fig.suptitle("Min WCRT & MS Dev: Naive vs. KD vs. Obstacle (Line)", fontsize=15)
        plt.tight_layout()
        plt.show()

    def plot_aspect_ratio_bar_triple(self, log_scale=False):
        """
        Bar plot for aspect ratio with 3 approaches side by side.
        """
        if (self.naive_aspect is None) or (self.kd_aspect is None) or (self.obs_aspect is None):
            print("Missing triple data for aspect ratio. Skipping.")
            return

        x = np.arange(len(self.depths))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - width, self.naive_aspect, width, color="tab:blue",  label='Naive KD')
        ax.bar(x, self.kd_aspect, width, color='tab:orange', label='Perimeter-Based KD')
        ax.bar(x + width, self.obs_aspect, width, color='tab:green', label='Obstacle-Aware')
        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in self.depths])
        # ax.set_title("Aspect Ratio", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Aspect Ratio", fontsize=12)
        ax.axhline(y=0.5, color='red', linestyle='--', label='Aspect Ratio = 0.5')

        ax.legend(fontsize=11)
        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_aspect_ratio_line_triple(self, log_scale=False):
        """
        Line plot for aspect ratio with 3 approaches.
        """
        if (self.naive_aspect is None) or (self.kd_aspect is None) or (self.obs_aspect is None):
            print("Missing triple data for aspect ratio. Skipping.")
            return

        x = self.depths
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, self.naive_aspect, 'o--', color='tab:blue', label='Naive KD')
        ax.plot(x, self.kd_aspect, 's--', color='tab:orange', label='Perimeter KD')
        ax.plot(x, self.obs_aspect, '^--', color='tab:green', label='Obstacle-Aware')
        ax.set_title("Aspect Ratio", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Aspect Ratio", fontsize=12)
        ax.legend(fontsize=11)
        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

    def plot_execution_time_bar_triple(self, log_scale=False):
        """
        Plots a single bar chart with 3 bars per depth, showing each approach's
        total runtime. If any approach is missing execution time, we skip.
        """
        # Check if we have data
        if (self.naive_exec is None) or (self.kd_exec is None) or (self.obs_exec is None):
            print("Missing execution time data for one or more approaches. Skipping execution time bar plot.")
            return

        x = np.arange(len(self.depths))
        width = 0.2

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(x - width, self.naive_exec, width, color='tab:blue', label='Naive KD')
        ax.bar(x, self.kd_exec, width, color='tab:orange',  label='Perimeter-Based KD')
        ax.bar(x + width, self.obs_exec, width, label='Obstacle-Aware')

        ax.set_xticks(x)
        ax.set_xticklabels([str(d) for d in self.depths])
        # ax.set_title("Execution Time", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.legend(fontsize=11)

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()


    def plot_execution_time_line_triple(self, log_scale=False):
        """
        Line plot for each approach's execution time over depth.
        """
        if (self.kd_exec is None) or (self.obs_exec is None):
            print("Missing execution time data for one or more approaches. Skipping execution time line plot.")
            return

        x = self.depths
        fig, ax = plt.subplots(figsize=(10, 6))

        # ax.plot(x, self.naive_exec, 'o--', color='tab:blue', label='Naive KD')
        ax.plot(x, self.kd_exec, 's--', color='tab:orange', label='Perimeter-Based KD')
        ax.plot(x, self.obs_exec, '^--', color='tab:green', label='Obstacle-Aware')

        # ax.set_title("Tree Construction Time", fontsize=14)
        ax.set_xlabel("Depth", fontsize=12)
        ax.set_ylabel("Time (seconds)", fontsize=12)
        ax.legend(fontsize=11)

        if log_scale:
            ax.set_yscale('log')

        plt.tight_layout()
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_aspect_and_execution(depths, kd_aspect=None, obs_aspect=None, kd_exec=None, obs_exec=None, log_scale=False):
    """
    Plot aspect ratio (bar chart) and execution time (line plot) in a single figure with two columns.

    Parameters:
        depths (list): Depth values.
        kd_aspect (list): Aspect ratio values for Perimeter-Based KD.
        obs_aspect (list): Aspect ratio values for Obstacle-Aware.
        kd_exec (list): Execution time values for Perimeter-Based KD.
        obs_exec (list): Execution time values for Obstacle-Aware.
        log_scale (bool): Whether to use logarithmic scale for the y-axis.
    """
    if kd_aspect is None or obs_aspect is None or kd_exec is None or obs_exec is None:
        print("Required data not provided. Please ensure all inputs are non-empty lists.")
        return

    # Create the figure with two columns
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), constrained_layout=True)

    # Plot for Aspect Ratio (Bar Chart)
    x = np.arange(len(depths))
    width = 0.35

    axes[0].bar(x - width / 2, kd_aspect, width, color='tab:orange', label='Perimeter-Based')
    axes[0].bar(x + width / 2, obs_aspect, width, color='tab:green', label='Obstacle-Aware')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(d) for d in depths])
    axes[0].set_xlabel("Depth", fontsize=12)
    axes[0].set_ylabel("Aspect Ratio", fontsize=12)
    axes[0].axhline(y=0.5, color='red', linestyle='--', label='Aspect Ratio = 0.5')
    axes[0].legend(fontsize=11)
    # axes[0].set_title("Aspect Ratio", fontsize=14)

    # Plot for Execution Time (Line Plot)
    axes[1].plot(depths, kd_exec, 's--', color='tab:orange', label='Perimeter-Based KD')
    axes[1].plot(depths, obs_exec, '^--', color='tab:green', label='Obstacle-Aware')
    axes[1].set_xlabel("Depth", fontsize=12)
    axes[1].set_ylabel("Time (seconds)", fontsize=12)
    axes[1].legend(fontsize=11)
    # axes[1].set_title("Execution Time", fontsize=14)

    if log_scale:
        # axes[0].set_yscale('log')
        axes[1].set_yscale('log')

    # Display the plots
    plt.show()


# Example usage (if you run this file as a script):
if __name__ == "__main__":
    import random

    # Provide output data in list forms corresponding to depths

    # Synthetic Data
    # depths = [2, 3, 4, 5, 6]
    # # # # Iowa Data
    depths = [2, 4, 6, 8, 10]
    # #
    # Naive_KD-tree
    naive_kd_avg_wcrt = [479778, 189156, 82572, 38119, 18229]
    naive_kd_min_wcrt = [426361, 106002, 49657, 12734, 2115]
    naive_kd_max_wcrt = [582837, 307329, 132800, 69155, 35202]
    naive_kd_msdev = [60813, 45197, 18402, 8310, 3782]
    naive_execution_time = [0.027, 0.056, 0.10, 0.223, 0.552]
    naive_kd_aspect_ratio = [0.65, 0.664, 0.662, 0.664, 0.66]


    # # # KD-tree metrics
    kd_avg_wcrt      = [476276, 194877, 89597, 44391, 22599]
    kd_min_wcrt  = [450330,  102360,  18991,  3800,  213]
    kd_max_wcrt  = [503059,  297895,  204186,  191212,  189118]
    kd_msdev     = [19571,   56679,   49800,   43788,   35019]
    kd_execution_time = [326.14, 414.50, 468, 476, 491]
    kd_aspect_ratio = [0.552,   0.56,   0.413,   0.355,   0.282]
    #
    #

    # # Obstacle-Aware example metrics
    obs_avg_wcrt     = [498160, 197294, 81368, 37405, 17387]
    obs_min_wcrt = [475175,  150920,  46796,  10697,  3017]
    obs_max_wcrt = [525500,  223560,  112293,  56643,  31450]
    obs_msdev    = [20424,   21687,   18186,   10919,   6379]
    # obs_variance = [417170830,  470339425,  330751926,  119244725,  40693868]
    # obs_range    = [50,   21687,   18186,   10919,   6403]
    obs_execution_time   = [1385,   1416,   1682,   1760,  1947]
    obs_aspect_ratio = [0.562,  0.568,  0.662,   0.62,   0.657]

    #  ========================================================
    #Synthetic data 5% obstacle coverage

    # Naive_KD-tree
    # naive_kd_avg_wcrt = [149.61, 95.35, 55.08, 37.81, 22.61]
    # naive_kd_min_wcrt = [125.26, 78.34, 43.09, 27.95, 17.68]
    # naive_kd_max_wcrt = [181.48, 118.48, 86.12, 60.94, 39.25]
    # naive_kd_msdev = [21.386, 14.364, 11.208, 6.729, 4.568]
    # naive_execution_time = [0.0126, 0.014, 0.020, 0.031, 0.04]
    # naive_kd_aspect_ratio = [1 , 0.5, 1, 0.5, 1]
    #
    # # KD-tree
    # kd_avg_wcrt = [149.92, 95.78, 55.77, 38.71, 24]
    # kd_min_wcrt = [139.61, 81.41, 37.13, 19.59, 7.48]
    # kd_max_wcrt = [157.43, 107.15, 69.05, 52.33, 44.03]
    # kd_msdev = [7.559, 9.843, 7.556, 8.382, 9.084]
    # kd_execution_time = [2.05, 2.65, 3.08, 3.416, 5.091]
    # kd_aspect_ratio = [0.832, 0.517, 0.695, 0.491, 0.525]
    #
    # # Obstacle-Aware
    # obs_avg_wcrt = [149.7, 95.55, 55.39, 37.5, 23.5]
    # obs_min_wcrt = [149.2, 90.96, 52.58, 31.4, 18.29]
    # obs_max_wcrt = [150.3, 99.82, 58.43, 42.39, 32.81]
    # obs_msdev = [0.54, 3.42, 2.17, 3.23, 2.854]
    # obs_execution_time   = [9, 12.4, 15.39, 18.28, 21.06]
    # obs_aspect_ratio = [0.889, 0.523, 0.792, 0.563, 0.682]


    #  ========================================================
    #Synthetic data 10% obstacle coverage

    # # Naive_KD-tree
    # naive_kd_avg_wcrt = [211.7, 126.4, 70.61, 45.57, 26.49]
    # naive_kd_min_wcrt = [194.31, 115.5, 53.93, 27.95, 17.68]
    # naive_kd_max_wcrt = [244.12, 161.0, 102.88, 68.99, 43.66]
    # naive_kd_msdev = [19.207, 13.871, 10.379, 8.649, 5.384]
    # naive_execution_time = [0.042, 0.025, 0.038, 0.115, 0.078]
    # naive_kd_aspect_ratio = [1 , 0.5, 1, 0.5, 1]
    #
    # # KD-tree
    # kd_avg_wcrt = [211.76, 126.56, 70.9, 46.04, 27.28]
    # kd_min_wcrt = [207.35, 113.36, 53.54, 28.93, 11.16]
    # kd_max_wcrt = [216.34, 132.49, 85.53, 56.52, 42.21]
    # kd_msdev = [3.863559, 7.445, 7.169, 6.833, 6.791]
    # kd_execution_time = [5.27, 7.210, 7.753, 9.94, 10.3]
    # kd_aspect_ratio = [0.934, 0.502, 0.81, 0.616, 0.445]
    #
    # # Obstacle-Aware
    # obs_avg_wcrt =            [211.75, 126.49, 70, 45.3, 27.25]
    # obs_min_wcrt =            [210.54, 124.16, 67.08, 40.51, 22.61]
    # obs_max_wcrt =            [212.97, 128.38, 72.09, 50.89, 36.87]
    # obs_msdev =               [1.215, 1.56, 1.644, 2.63, 3.13]
    # obs_execution_time   =    [18.45, 25.72, 29.85, 34.05, 38.8]
    # obs_aspect_ratio =        [0.93, 0.502, 0.86, 0.564, 0.706]


    #============10==========
    # Naive_KD-tree
    # naive_kd_avg_wcrt = [211.7, 126.4, 70.61, 45.57, 26.49]
    # naive_kd_min_wcrt = [194.31, 115.5, 53.93, 27.95, 17.68]
    # naive_kd_max_wcrt = [244.12, 161.0, 102.88, 68.99, 43.66]
    # naive_kd_msdev = [19.207, 13.871, 10.379, 8.649, 5.384]
    # naive_execution_time = [0.042, 0.025, 0.038, 0.115, 0.078]
    # naive_kd_aspect_ratio = [1 , 0.5, 1, 0.5, 1]
    #
    # # KD-tree
    # kd_avg_wcrt = [211.76, 126.56, 70.9, 46.04, 27.28]
    # kd_min_wcrt = [207.35, 113.36, 53.54, 28.93, 11.16]
    # kd_max_wcrt = [216.34, 132.49, 85.53, 56.52, 42.21]
    # kd_msdev = [3.863559, 7.445, 7.169, 6.833, 6.791]
    # kd_execution_time = [5.27, 7.210, 7.753, 9.94, 10.3]
    # kd_aspect_ratio = [0.934, 0.502, 0.81, 0.616, 0.445]
    #
    # # Obstacle-Aware
    # obs_avg_wcrt =            [211.75, 126.49, 70, 45.3, 27.25]
    # obs_min_wcrt =            [210.54, 124.16, 67.08, 40.51, 22.61]
    # obs_max_wcrt =            [212.97, 128.38, 72.09, 50.89, 36.87]
    # obs_msdev =               [1.215, 1.56, 1.644, 2.63, 3.13]
    # obs_execution_time   =    [18.45, 25.72, 29.85, 34.05, 38.8]
    # obs_aspect_ratio =        [0.93, 0.502, 0.86, 0.564, 0.706]


    plotter = WCRTComparisonPlotter(
        depths=depths,

        naive_avg_wcrt=naive_kd_avg_wcrt,
        naive_min_wcrt=naive_kd_min_wcrt,
        naive_max_wcrt=naive_kd_max_wcrt,
        naive_msdev=naive_kd_msdev,
        naive_aspect_ratio=naive_kd_aspect_ratio,
        naive_exec_time=naive_execution_time,

        kd_avg_wcrt=kd_avg_wcrt,
        kd_min_wcrt=kd_min_wcrt,
        kd_max_wcrt=kd_max_wcrt,
        kd_msdev=kd_msdev,
        kd_aspect_ratio=kd_aspect_ratio,
        kd_exec_time=kd_execution_time,

        obs_avg_wcrt=obs_avg_wcrt,
        obs_min_wcrt=obs_min_wcrt,
        obs_max_wcrt=obs_max_wcrt,
        obs_msdev=obs_msdev,
        obs_exec_time=obs_execution_time,
        obs_aspect_ratio=obs_aspect_ratio
    )

    # 1) Max & Avg WCRT => bar
    # plotter.plot_avg_and_max_wcrt_bar(log_scale=False)
    # # 2) Max & Avg WCRT => line
    # plotter.plot_avg_and_max_wcrt_line(log_scale=False)
    #
    # # 3) Min & MS dev => bar
    # plotter.plot_min_and_msdev_bar(log_scale=False)
    # # 4) Min & MS dev => line
    # plotter.plot_min_and_msdev_line(log_scale=False)
    #
    # 5) aspect ratio => bar
    plotter.plot_aspect_ratio_bar(log_scale=False)

    # # 6) aspect ratio => line
    # plotter.plot_aspect_ratio_line(log_scale=False)

    plotter.plot_avg_and_max_wcrt_bar_triple()
    # plotter.plot_avg_and_max_wcrt_line_triple()
    #
    plotter.plot_min_and_msdev_bar_triple()
    # plotter.plot_min_and_msdev_line_triple()
    #
    # plotter.plot_aspect_ratio_bar_triple(log_scale=False)
    # # plotter.plot_aspect_ratio_line_triple(log_scale=True)
    #
    # # plotter.plot_execution_time_bar_triple(log_scale=False)
    plotter.plot_execution_time_line_triple(log_scale=True)
    # plot_aspect_and_execution(depths, kd_aspect_ratio, obs_aspect_ratio, kd_execution_time, obs_execution_time, log_scale=True)
