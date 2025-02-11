# import logging
from src.obstacle_aware_divider import ObstacleAwareDivider
from src.strip_perimeter import Strip
#
class OptimalAxisSelection:
    """
    Simplified class that *only* evaluates NWCRT and uses MSDU as a tie-breaker.
    All other metrics and unnecessary computations are commented out for efficiency.
    """

    _METRIC_IS_MINIMIZED = {
        "NWCRT": True  # smaller NWCRT is better
    }

    def __init__(self, region, obstacles, user_metric=None, numerical_method='newton',
                 tie_threshold=1e-2):
        """
        We keep the essential parameters and default to NWCRT if user_metric is missing or invalid.
        """
        self.region = region
        self.obstacles = obstacles
        self.numerical_method = numerical_method
        self.tie_threshold = tie_threshold

        # Force user_metric to "NWCRT" if not provided or invalid:
        self.user_metric = user_metric if user_metric == "NWCRT" else "NWCRT"

    def evaluate_axis(self, axis):
        """
        1) Build Strip manager + ObstacleAwareDivider.
        2) Find division point, split region.
        3) Calculate NWCRT & MSDU for tie-breaking.
        4) Return metrics dict (including division_point & subregions).
        """
        strip_manager = Strip(self.region, self.obstacles, axis=axis)
        divider = ObstacleAwareDivider(strip_manager, method=self.numerical_method)

        division_point = divider.find_optimal_division_point()
        (R_left, left_obstacles), (R_right, right_obstacles) = divider.divide_region(division_point)

        left_sm = Strip(R_left, left_obstacles, axis=axis)
        right_sm = Strip(R_right, right_obstacles, axis=axis)

        wcrt_left = left_sm.calculate_region_wcrt()
        wcrt_right = right_sm.calculate_region_wcrt()

        sum_wcrt = wcrt_left + wcrt_right
        diff_wcrt = abs(wcrt_left - wcrt_right)

        metrics = {}
        metrics["NWCRT"] = diff_wcrt / sum_wcrt if sum_wcrt > 1e-9 else 0.0

        # For tie-breaking (MSDU):
        sq_left = self._square_measure(R_left)
        sq_right = self._square_measure(R_right)
        _, msdu = self.calculate_squareness_metrics(sq_left, sq_right)
        metrics["_MSDU"] = msdu

        # Include geometry so we can re-use them
        metrics["_division_point"] = division_point
        metrics["_subregion_left"] = (R_left, left_obstacles)
        metrics["_subregion_right"] = (R_right, right_obstacles)
        return metrics

    def _square_measure(self, polygon):
        if polygon.is_empty:
            return 1.0
        minx, miny, maxx, maxy = polygon.bounds
        w = maxx - minx
        h = maxy - miny
        if w < 1e-9 and h < 1e-9:
            return 1.0
        if w < 1e-9 or h < 1e-9:
            return 0.0
        return w / h

    def calculate_squareness_metrics(self, sq_left, sq_right, eps=1e-9):
        msdu = 1 / ((0.5 * ((sq_left - 1.0)**2 + (sq_right - 1.0)**2)) + eps)
        return None, msdu

    def select_best_axis(self):
        """
        Evaluate 'x' and 'y' => NWCRT_x, NWCRT_y, then pick axis with smaller NWCRT.
        If tie => pick axis with larger MSDU.
        We also return the subregions for the chosen axis to avoid repeated computations.
        """
        metrics_x = self.evaluate_axis('x')
        metrics_y = self.evaluate_axis('y')

        value_x = metrics_x["NWCRT"]
        value_y = metrics_y["NWCRT"]

        diff = abs(value_x - value_y)
        if diff <= self.tie_threshold:
            # tie => compare MSDU
            if metrics_x["_MSDU"] > metrics_y["_MSDU"]:
                best_axis = 'x'
                chosen_metrics = metrics_x
            else:
                best_axis = 'y'
                chosen_metrics = metrics_y
        else:
            # For NWCRT => smaller is better
            if value_x < value_y:
                best_axis = 'x'
                chosen_metrics = metrics_x
            else:
                best_axis = 'y'
                chosen_metrics = metrics_y

        # Extract subregions & division point for the best axis
        best_div_point = chosen_metrics["_division_point"]
        best_sub_left = chosen_metrics["_subregion_left"]  # (R_left, left_obs)
        best_sub_right = chosen_metrics["_subregion_right"]  # (R_right, right_obs)

        # Return the axis, the full metrics for both axes if desired,
        # plus the actual geometry for partition.
        overall_metrics = {"x": metrics_x, "y": metrics_y}

        return best_axis, overall_metrics, best_div_point, best_sub_left, best_sub_right

#
