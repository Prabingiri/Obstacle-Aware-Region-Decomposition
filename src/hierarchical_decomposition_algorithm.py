
import logging
import argparse
from shapely.validation import make_valid
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union

from src.obstacle_aware_divider import ObstacleAwareDivider
from src.optimal_axis_selection import OptimalAxisSelection
from src.strip_perimeter import Strip

#####################
# USER-ADJUSTABLE:
DRONE_THRESHOLD = 5.0
COVERAGE_RATIO_STOP = 0.90
MIN_LARGEST_CONNECTED_SPACE = DRONE_THRESHOLD
#####################


class HierarchicalDecomposition:
    """
    Recursively decompose a region (with obstacles) into valid, navigable subregions.

    Key Stopping Conditions (unchanged):
      1) If obstacle coverage >= 90%: --change according to requirement
         - If leftover free area < DRONE_THRESHOLD => store region (stop).
         - Else if the largest connected free space >= DRONE_THRESHOLD => subdivide more.
      2) If depth >= max_depth => store region.
      3) If region is invalid => track_back => restore the parent region.
      4) Fallback axis if the chosen axis fails to produce valid child partitions.
      5) track_back ensures we do NOT lose the region if it was the last valid representation.
    """

    def __init__(
        self,
        region,
        obstacles,
        max_depth=3,
        metrics="TotalWCRT",
        numerical_method="newton",
        min_dimension_threshold=1e-3,
        check_connectivity=False,
        allow_fallback_axis=True,
        mode="track_back",  # originally 'track_back' or 'force_depth', but we only keep track_back now
    ):
        self.region = self._validate_geometry(region)
        self.obstacles = [self._validate_geometry(obs) for obs in obstacles]
        self.max_depth = max_depth
        self.metrics = metrics
        self.numerical_method = numerical_method
        self.min_dimension_threshold = min_dimension_threshold
        self.check_connectivity = check_connectivity
        self.allow_fallback_axis = allow_fallback_axis
        self.mode = mode  # We'll keep this param but effectively ignore any "force_depth" usage.

        # Each final partition => (region, obstacles, axis_stack, is_valid)
        self.partitions = []
        self.axis_stack = []

    def _validate_geometry(self, geom):
        """Ensure geometry is valid; if not, fix with make_valid."""
        if not geom.is_valid:
            # logging.warning("[_validate_geometry] Invalid geometry. Attempting make_valid.")
            geom = make_valid(geom)
        return geom

    def run(self):
        """Top-level entry: begin recursive decomposition."""
        # logging.info("[HierarchicalDecomposition] Starting decomposition...")
        produced_any_valid = self._decompose(self.region, self.obstacles, depth=0)

        # If track_back mode and no valid child => store entire region
        if self.mode == "track_back" and not produced_any_valid:
            # logging.warning("[HierarchicalDecomposition] No valid partitions => storing top-level region.")
            self.partitions.append((self.region, self.obstacles, list(self.axis_stack), True))

        # logging.info(f"[HierarchicalDecomposition] Done. Created {len(self.partitions)} partitions.")
        return self.partitions

    def _decompose(self, region, obstacles, depth) -> bool:
        """
        Recursively decompose 'region' with 'obstacles' at recursion 'depth'.
        Returns True if at least one valid final partition was produced, else False => track_back.
        """
        # Basic checks
        if not region or region.is_empty:
            # logging.warning("[_decompose] Region empty => invalid.")
            return False  # track_back => return False

        region_area = region.area
        obs_area = sum(ob.area for ob in obstacles)
        coverage_ratio = obs_area / region_area if region_area > 1e-12 else 1.0
        free_area = region_area - obs_area

        largest_hole_area = self._compute_largest_free_space(region, obstacles)
        # logging.debug(f"region_area={region_area}, coverage={coverage_ratio*100:.1f}%, largest_hole={largest_hole_area}")

        # If coverage≥90% and leftover free < DRONE_THRESHOLD and largest hole < DRONE_THRESHOLD => store region
        if coverage_ratio >= COVERAGE_RATIO_STOP:
            if free_area < DRONE_THRESHOLD and largest_hole_area < DRONE_THRESHOLD:
                # logging.info(f"[STOP] coverage≥{COVERAGE_RATIO_STOP*100:.1f}% & free<DRONE, storing region.")
                is_ok = self._is_subregion_valid(region, obstacles)
                self.partitions.append((region, obstacles, list(self.axis_stack), is_ok))
                return True

        # If depth >= max_depth => store
        if depth >= self.max_depth:
            is_ok = self._is_subregion_valid(region, obstacles)
            # logging.info(f"[STOP] Reached max depth => storing partition (is_valid={is_ok}).")
            self.partitions.append((region, obstacles, list(self.axis_stack), is_ok))
            return is_ok

        # Subregion validity check
        if not self._is_subregion_valid(region, obstacles):
            # logging.warning("[_decompose] Subregion invalid => track_back => returning False.")
            return False

        # 1) Select the best axis (and get subregions) => no second partition call needed later
        try:
            axis_selector = OptimalAxisSelection(
                region, obstacles,
                user_metric=self.metrics,
                numerical_method=self.numerical_method
            )
            # Must return: best_axis, overall_metrics, best_div_pt, subL, subR
            best_axis, _, best_div_pt, subL, subR = axis_selector.select_best_axis()
            # subL => (R_left, left_obs), subR => (R_right, right_obs)
        except Exception as e:
            # logging.error(f"[AxisSelection] Error: {e}")
            return False  # track_back => return False

        # 2) Try partition with best_axis
        if self._attempt_partition(region, obstacles, best_axis, depth, best_div_pt, subL, subR):
            return True
        else:
            # 3) Fallback axis if allowed
            if self.allow_fallback_axis:
                fallback_axis = "y" if best_axis == "x" else "x"
                # logging.info(f"[Fallback] Trying fallback axis={fallback_axis} at depth={depth}")
                # We must re-run axis selection for fallback, or forcibly partition:
                try:
                    axis_selector_fallback = OptimalAxisSelection(
                        region, obstacles,
                        user_metric=self.metrics,
                        numerical_method=self.numerical_method
                    )
                    fb_axis, _, fb_div_pt, fb_subL, fb_subR = axis_selector_fallback.select_best_axis()
                    if fb_axis == best_axis:
                        # The same axis was chosen => no difference
                        return False
                    return self._attempt_partition(region, obstacles, fb_axis, depth, fb_div_pt, fb_subL, fb_subR)
                except Exception:
                    return False

            # logging.warning("[_decompose] Partition (both axes) failed => returning False.")
            return False

    def _attempt_partition(self, region, obstacles, axis, depth, division_point, subL, subR) -> bool:
        """
        Attempt the partition along 'axis' using the ALREADY known division_point
        """
        self.axis_stack.append(axis)

        # Check for degenerate cut
        minx, miny, maxx, maxy = region.bounds
        if axis == "x":
            if division_point <= minx or division_point >= maxx:
                # logging.warning("[_attempt_partition] Degenerate partition along x => fails.")
                self.axis_stack.pop()
                return False
        else:
            if division_point <= miny or division_point >= maxy:
                # logging.warning("[_attempt_partition] Degenerate partition along y => fails.")
                self.axis_stack.pop()
                return False

        (R_left, left_obs) = subL
        (R_right, right_obs) = subR

        # Evaluate validity
        left_ok = self._is_subregion_valid(R_left, left_obs)
        right_ok = self._is_subregion_valid(R_right, right_obs)
        # logging.debug(f"[_attempt_partition] left_ok={left_ok}, right_ok={right_ok}")

        if not left_ok and not right_ok:
            # logging.warning("[_attempt_partition] Both children invalid => partition fails.")
            self.axis_stack.pop()
            return False

        produced_any = False

        # Recurse on left if valid
        if left_ok:
            if self._decompose(R_left, left_obs, depth + 1):
                produced_any = True

        # Recurse on right if valid
        if right_ok:
            if self._decompose(R_right, right_obs, depth + 1):
                produced_any = True

        self.axis_stack.pop()
        return produced_any

    def _is_subregion_valid(self, region, obstacles):
        """Check geometry validity, dimension threshold, coverage ratio, connectivity, etc."""
        if not region or region.is_empty:
            return False
        if not region.is_valid:
            return False

        minx, miny, maxx, maxy = region.bounds
        width = maxx - minx
        height = maxy - miny
        if width < self.min_dimension_threshold or height < self.min_dimension_threshold:
            return False

        region_area = region.area
        obs_area_sum = sum(ob.area for ob in obstacles)
        if obs_area_sum >= region_area - 1e-9:
            return False

        coverage_ratio = obs_area_sum / region_area if region_area > 1e-12 else 1.0
        if coverage_ratio >= COVERAGE_RATIO_STOP:
            largest_hole = self._compute_largest_free_space(region, obstacles)
            if largest_hole < DRONE_THRESHOLD:
                return False

        if self.check_connectivity:
            if not self._has_single_connected_free_space(region, obstacles):
                return False

        return True

    def _compute_largest_free_space(self, region, obstacles):
        """Return area of the largest free-space polygon within the region."""
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
        except Exception as e:
            # logging.error(f"[_compute_largest_free_space] Error: {e}")
            return 0.0

    def _has_single_connected_free_space(self, region, obstacles):
        """Check if there's at least one free polygon >= DRONE_THRESHOLD."""
        largest_area = self._compute_largest_free_space(region, obstacles)
        return largest_area >= DRONE_THRESHOLD
