import matplotlib.pyplot as plt
import numpy as np
import random


def plot_max_wcrt_subplots(strategies_data, depths, obstacle_variations, use_log=False):
    """
    Plots Maximum WCRT vs. Depth in a 2x3 grid of subplots for each obstacle percentage,

    If use_log=True, the Y-axis will be set to log scale.
    """
    strategy_colors = {
        "Naive KD": "tab:blue",
        "Perimeter-Based KD": "tab:orange",
        "Obstacle-Aware": "tab:green"
    }

    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=False)
    axs = axs.flatten()

    for idx, obs_perc in enumerate(obstacle_variations):
        ax = axs[idx]
        for strategy, obs_data in strategies_data.items():
            if obs_perc in obs_data:
                data = obs_data[obs_perc]
                color = strategy_colors.get(strategy, "black")
                ax.plot(depths,
                        data["max_wcrt"],
                        marker='o',
                        markersize=6,
                        linestyle='-',
                        color=color,
                        label=strategy)
        ax.set_title(f"Obstacles {obs_perc}%", fontsize=12)
        ax.grid(True)
        ax.legend(fontsize=8)

        # If log scale is requested, set it here
        if use_log:
            ax.set_yscale('log')

    # Remove unused subplots if any
    for j in range(len(obstacle_variations), len(axs)):
        fig.delaxes(axs[j])

    # fig.suptitle("Maximum WCRT vs. Depth Across Strategies", fontsize=16)
    fig.supxlabel("Depth", fontsize=14)
    fig.supylabel("Maximum WCRT", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_std_dev_subplots(strategies_data, depths, obstacle_variations, use_log=False):
    """
    Plots Standard Deviation of WCRT vs. Depth in a 2x3 grid of subplots for each obstacle percentage.
    If use_log=True, the Y-axis will be set to log scale.

    Modified so that only a single legend is displayed for the entire figure.
    """
    import matplotlib.pyplot as plt

    strategy_colors = {
        "Naive KD": "tab:blue",
        "Perimeter-Based KD": "tab:orange",
        "Obstacle-Aware": "tab:green"
    }

    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True, sharey=False)
    axs = axs.flatten()

    for idx, obs_perc in enumerate(obstacle_variations):
        # Stop if we have more obstacle perc than subplots
        if idx >= len(axs):
            break

        ax = axs[idx]
        for strategy, obs_data in strategies_data.items():
            if obs_perc in obs_data:
                data = obs_data[obs_perc]
                color = strategy_colors.get(strategy, "black")

                # Only label in the first subplot, to avoid repeating legends
                label = strategy if idx == 0 else ""

                ax.plot(
                    depths,
                    data["std_dev"],
                    marker='o',
                    markersize=6,
                    linestyle='-',
                    color=color,
                    label=label
                )

        ax.set_title(f"Obstacles {obs_perc}%", fontsize=12)
        ax.grid(True)

        # If log scale is requested
        if use_log:
            ax.set_yscale('log')

    # Remove any unused subplots
    for j in range(len(obstacle_variations), len(axs)):
        fig.delaxes(axs[j])

    # Create a single legend from the handles/labels of the first subplot
    handles, labels = axs[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc='upper center', ncol=len(strategy_colors), fontsize=8)

    fig.supxlabel("Depth", fontsize=14)
    fig.supylabel("WCRT Standard Deviation", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.92]) 
    plt.show()


def plot_max_wcrt_bar_charts(strategies_data, depths, obstacle_variations, use_log=False):
    """
    Creates a 2x3 grid of bar-chart subplots for Maximum WCRT.
    Each subplot now uses its own y-axis label (no shared Y-axis).
    If use_log=True, sets the Y-axis to log scale.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Fixed colors for each strategy
    strategy_colors = {
        "Naive KD": "tab:blue",
        "Perimeter-Based KD": "tab:orange",
        "Obstacle-Aware": "tab:green"
    }

    strategies = list(strategies_data.keys())
    n_strategies = len(strategies)
    n_depths = len(depths)

    fig, axs = plt.subplots(3, 2, figsize=(10, 6))
    axs = axs.flatten()

    bar_width = 0.2
    x_positions = np.arange(n_depths)

    for idx, obs_perc in enumerate(obstacle_variations):
        if idx >= len(axs):
            break
        ax = axs[idx]
        for s_idx, strategy in enumerate(strategies):
            obs_data = strategies_data[strategy].get(obs_perc)
            if obs_data:
                offset = (s_idx - (n_strategies - 1) / 2) * bar_width
                positions = x_positions + offset
                color = strategy_colors.get(strategy, "gray")

                # Label only in the first subplot so we don't repeat legends
                label = strategy if idx == 0 else ""

                ax.bar(
                    positions,
                    obs_data["max_wcrt"],
                    width=bar_width,
                    color=color,
                    label=label,
                    alpha=0.8
                )

        ax.set_title(f"Obstacles {obs_perc}%", fontsize=10)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(d) for d in depths], fontsize=8)
        ax.grid(True, axis="y")

        # Give each subplot its own y-axis label
        ax.set_ylabel("Maximum WCRT", fontsize=9)

        # If log scale is requested, set it
        if use_log:
            ax.set_yscale('log')

    # Remove unused subplots
    for j in range(len(obstacle_variations), len(axs)):
        fig.delaxes(axs[j])

    # Create a single legend from the handles/labels of the first subplot
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(strategies), fontsize=9)

    # fig.suptitle("Maximum WCRT (Bar Charts) Across Depths & Obstacles", fontsize=12)
    fig.supxlabel("Depth", fontsize=10)

    plt.tight_layout()
    plt.show()


def plot_max_wcrt_boxplots(strategies_data, depths, obstacle_variations, use_log=False):
    """
    Creates a 2x3 grid of boxplot subplots for Maximum WCRT.
    Each subplot corresponds to one obstacle percentage, and each strategy appears as one box
    (with the data from different depths as the distribution).

    If use_log=True, the Y-axis is set to log scale (helpful if there's a wide range of WCRTs).
    """
    strategies = list(strategies_data.keys())
    fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharey=False)
    axs = axs.flatten()

    for idx, obs_perc in enumerate(obstacle_variations):
        if idx >= len(axs):
            break
        ax = axs[idx]

        data_for_boxplot = []
        labels_for_boxplot = []

        for strategy in strategies:
            obs_data = strategies_data[strategy].get(obs_perc)
            if obs_data:
                data_for_boxplot.append(obs_data["max_wcrt"])
                labels_for_boxplot.append(strategy)
            else:
                # If no data, append empty data to keep alignment
                data_for_boxplot.append([])
                labels_for_boxplot.append(strategy)

        # Plot the boxplot
        # positions = range(1, len(strategies)+1) to place them across the x-axis
        positions = range(1, len(strategies) + 1)
        bp = ax.boxplot(data_for_boxplot, positions=positions, patch_artist=True)

        # Optional: color each box
        colors = ["blue", "orange", "green", "red", "purple", "gray"]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels_for_boxplot, rotation=15, fontsize=9)
        ax.set_title(f"Obstacle {obs_perc}%", fontsize=12)
        ax.grid(True, axis="y")

        # Y-axis label (only if it's in the left column, to avoid clutter)
        if idx % 2 == 0:
            ax.set_ylabel("Max WCRT", fontsize=10)

        if use_log:
            ax.set_yscale('log')

    # Remove extra subplots if any
    for j in range(len(obstacle_variations), len(axs)):
        fig.delaxes(axs[j])

    # fig.suptitle("Maximum WCRT Boxplots Across Strategies & Obstacle %", fontsize=16)
    fig.supxlabel("Strategies", fontsize=14)
    fig.supylabel("Maximum WCRT", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# Main block with synthetic data
if __name__ == "__main__":
    # Define parameters
    strategies = ["Naive KD", "Perimeter-Based KD", "Obstacle-Aware"]
    obstacle_variations = [5, 10, 15, 20, 25, 30]
    depths = [2, 3, 4, 5, 6]

    #Data Input
    strategies_data = {
        "Naive KD": {
            5: {
                "max_wcrt": [181.48, 118.48, 86.12, 60.94, 39.25],
                "std_dev": [21.386, 14.364, 11.208, 6.729, 4.568]
            },
            10: {
                "max_wcrt": [244.12, 161.0, 102.88, 68.99, 43.66],
                "std_dev": [19.207, 13.871, 10.379, 8.649, 5.384]
            },
            15: {
                "max_wcrt": [274.75, 198.4, 121.69, 76.53, 42.17],
                "std_dev": [3.593, 23.3, 14.71, 5.626, 8.33]
            },
            20: {
                "max_wcrt": [364.43, 213.21, 131.14, 79.514, 49.72],
                "std_dev": [17.424, 15.98, 13.288, 8.917, 6.05]
            },
            25: {
                "max_wcrt": [468.93, 270.67, 144.56, 86.66, 55.17],
                "std_dev": [15.741, 14.084, 10.74, 7.27, 5.84]
            },
            30: {
                "max_wcrt": [573.69, 323.13, 174.89, 99.57, 61.66],
                "std_dev": [25.66, 17.97, 11.642, 8.04, 6.06]
            }

        },
        "Perimeter-Based KD": {
            5: {
                "max_wcrt": [157.43, 107.15, 69.05, 52.33, 44.03],
                "std_dev": [7.559, 9.843, 7.556, 8.382, 9.084]
            },
            10: {
                "max_wcrt":[216.34, 132.49, 85.53, 56.52, 42.21],
                "std_dev": [3.863559, 7.445, 7.169, 6.833, 6.791]
            },
            15: {
                "max_wcrt": [274.18, 161.07, 95.32, 64.66, 44.42],
                "std_dev": [1.784, 2.84, 4.656, 4.32, 5.315]
            },
            20: {
                "max_wcrt": [353.7, 201.2, 116.11, 75.69, 48.13],
                "std_dev": [4.56, 4.58, 5.38, 5.023, 5.007]
            },
            25: {
                "max_wcrt": [456, 251, 136.85, 83.22, 50.83],
                "std_dev": [6.034, 4.252, 2.69, 3.456, 3.683]
            },
            30: {
                "max_wcrt": [555.06, 302.78, 159.59, 95.64, 56.44],
                "std_dev": [5.308, 4.66, 3.298, 3.579, 3.8]
            }

        },
        "Obstacle-Aware": {
            5: {
                "max_wcrt": [149.7, 95.55, 55.39, 37.5, 23.5],
                "std_dev": [0.54, 3.42, 2.17, 3.23, 2.854]
            },
            10: {
                "max_wcrt": [212.97, 128.38, 72.09, 50.89, 36.87],
                "std_dev": [1.215, 1.56, 1.644, 2.63, 3.13]
            },
            15: {
                "max_wcrt": [271.41, 156.75, 89.75, 56.88, 38.43],
                "std_dev": [0.091, 0.385, 2.16, 2.581, 2.35]
            },
            20: {
                "max_wcrt": [347.93, 195.92, 106.26, 63.99, 42.66],
                "std_dev": [0.24, 1.101, 0.903, 1.2, 1.306]
            },
            25: {
                "max_wcrt": [448.63, 245.75, 131.46, 76.77, 43.11],
                "std_dev": [0.125, 0.747, 0.76, 1.032, 0.802]
            },
            30: {
                "max_wcrt": [548.92, 296.65, 156.35, 90.46, 49.27],
                "std_dev": [0.333, 1.306, 0.736, 1.274, 0.76]
            }
        }
    }


    # ======================
    strategies_data_no_uniform = {
        "Naive KD": {
            5: {
                "max_wcrt": [
                    214.11,
                    140.0,
                    97.48,
                    84.19,
                    57.31
                ],
                "std_dev": [
                    52.156,
                    30.409,
                    18.962,
                    13.136,
                    7.939
                ]
            },
            10: {
                "max_wcrt": [
                    305.83,
                    204.44,
                    125.69,
                    100.99,
                    65.92
                ],
                "std_dev": [
                    78.214,
                    46.591,
                    29.968,
                    20.004,
                    11.222
                ]
            },
            15: {
                "max_wcrt": [
                    445.56,
                    251.82,
                    176.52,
                    113.19,
                    73.41
                ],
                "std_dev": [
                    131.352,
                    71.857,
                    47.947,
                    27.873,
                    15.563
                ]
            },
            20: {
                "max_wcrt": [
                    647.35,
                    364.85,
                    238.04,
                    153.54,
                    104.06
                ],
                "std_dev": [
                    170.672,
                    98.171,
                    66.244,
                    39.346,
                    22.031
                ]
            },
            25: {
                "max_wcrt": [
                    1097.77,
                    578.69,
                    366.2,
                    241.38,
                    165.16
                ],
                "std_dev": [
                    315.93,
                    179.387,
                    112.077,
                    65.766,
                    37.448
                ]
            },
            30: {
                "max_wcrt": [
                    1162.76,
                    604.1,
                    380.79,
                    254.34,
                    169.72
                ],
                "std_dev": [
                    334.191,
                    190.742,
                    118.362,
                    68.641,
                    38.844
                ]
            }

        },
        "Perimeter-Based KD": {
            5: {
                "max_wcrt": [
                    170.02,
                    127.11,
                    100.62,
                    86.44,
                    77.03
                ],
                "std_dev": [
                    20.318,
                    25.055,
                    18.144,
                    20.367,
                    15.601
                ]
            },
            10: {
                "max_wcrt": [
                    237.62,
                    154.87,
                    107.93,
                    93.34,
                    70.56
                ],
                "std_dev": [
                    14.601,
                    19.482,
                    13.955,
                    15.706,
                    11.673
                ]
            },
            15: {
                "max_wcrt": [
                    331.45,
                    202.11,
                    129.8,
                    92.94,
                    72.42
                ],
                "std_dev": [
                    14.265,
                    18.639,
                    15.041,
                    15.462,
                    11.534
                ]
            },
            20: {
                "max_wcrt": [
                    472.66,
                    270.73,
                    156.73,
                    105.73,
                    74.65
                ],
                "std_dev": [
                    13.26,
                    16.674,
                    12.213,
                    13.348,
                    10.422
                ]
            },
            25: {
                "max_wcrt": [
                    764.36,
                    417.21,
                    231.9,
                    143.85,
                    91.39
                ],
                "std_dev": [
                    12.451,
                    16.281,
                    12.117,
                    13.178,
                    9.48
                ]
            },
            30: {
                "max_wcrt": [
                    803.57,
                    436.86,
                    241.83,
                    149.59,
                    92.73
                ],
                "std_dev": [
                    13.765,
                    16.863,
                    12.218,
                    13.456,
                    9.797
                ]
            }

        },
        "Obstacle-Aware": {
            5: {
                "max_wcrt": [
                    149.25,
                    108.33,
                    69.63,
                    53.63,
                    37.14
                ],
                "std_dev": [
                    0.795,
                    12.494,
                    6.765,
                    7.035,
                    6.146
                ]
            },
            10: {
                "max_wcrt": [
                    222.48,
                    143.76,
                    86.07,
                    64.6,
                    41.67
                ],
                "std_dev": [
                    0.542,
                    10.721,
                    5.249,
                    8.591,
                    6.049
                ]
            },
            15: {
                "max_wcrt": [
                    314.53,
                    188.94,
                    127.12,
                    75.22,
                    58.46
                ],
                "std_dev": [
                    0.65,
                    10.269,
                    11.006,
                    8.177,
                    6.496
                ]
            },
            20: {
                "max_wcrt": [
                    458.24,
                    259.6,
                    142.22,
                    91.35,
                    67.12
                ],
                "std_dev": [
                    0.208,
                    9.141,
                    5.021,
                    7.072,
                    7.226
                ]
            },
            25: {
                "max_wcrt": [
                    750.8,
                    405.96,
                    215.97,
                    130.28,
                    88.31
                ],
                "std_dev": [
                    0.751,
                    9.541,
                    5.158,
                    7.629,
                    6.631
                ]
            },
            30: {
                "max_wcrt": [
                    788.64,
                    424.96,
                    225.59,
                    135.27,
                    90.89
                ],
                "std_dev": [
                    0.964,
                    9.651,
                    5.261,
                    7.801,
                    6.675
                ]
            }
        }
    }



    # Plot subplots for Maximum WCRT and Standard Deviation
    plot_max_wcrt_subplots(strategies_data_no_uniform, depths, obstacle_variations, use_log=True)
    plot_std_dev_subplots(strategies_data_no_uniform, depths, obstacle_variations)

    # Now plot bar charts for Max WCRT
    plot_max_wcrt_bar_charts(strategies_data_no_uniform, depths, obstacle_variations)

    plot_max_wcrt_boxplots(strategies_data_no_uniform, depths, obstacle_variations, use_log=True)

