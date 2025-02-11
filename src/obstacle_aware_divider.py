
import math
import logging
from shapely import make_valid
from shapely.ops import unary_union
from shapely.geometry import Polygon, MultiPolygon, GeometryCollection, LineString, MultiLineString
from rtree import index

# from src.module4_1 import solve_for_root_brent, solve_for_root_newton_raphson, \
#     solve_for_root_with_defensive_newton_rhapson
from src.numerical_solution import solve_for_root_with_defensive_newton_rhapson, solve_for_root_brent


class ObstacleAwareDivider:
    def __init__(self, strip_processor, method='brent'):
        """
        Initialize with a Strip instance and a numerical root-finding method ('brent' or 'newton').
        """
        self.sp = strip_processor
        self.axis = self.sp.axis
        self.method = method

        # If you are not using an R-tree, we can comment this out:
        # self.spatial_index = None

        # Compute region-level diagonal + perimeter => WCRT
        self.d_R = self.sp.calculate_region_diagonal()
        self.P_R = self.sp.calculate_total_obstacle_perimeter()
        self.WCRT_total = self.d_R + 0.5 * self.P_R
        self.WCRT_target = self.sp.calculate_target_wcrt()

        # COMMENTING OUT debug logs:
        # logging.info(f"[ObstacleAwareDivider] WCRT Target: {self.WCRT_target:.4f}")
        # logging.info(f"[ObstacleAwareDivider] WCRT Total:  {self.WCRT_total:.4f}")

    # ---------------------------
    #   Finding the optimal cut
    # ---------------------------
    def find_optimal_division_point(self):
        """
        Finds the cut coordinate that balances WCRT between left/right (axis='x'/'y').
        """
        strip_interest = self.find_strip_of_interest()
        if strip_interest is None:
            raise RuntimeError("No strip of interest found (possibly no crossing).")

        coord_j_minus_1, coord_j, strip_geometry = strip_interest

        # Compute g at edges
        g_j = self.g(coord_j)
        # if abs(g_j) < 1e-6:
        #     logging.info(f"[find_optimal_division_point] near-zero => returning {coord_j:.4f}")
        #     return coord_j

        g_j_minus_1 = self.g(coord_j_minus_1)
        # logging.info(f"[find_optimal_division_point] g({coord_j_minus_1:.4f})={g_j_minus_1:.4f}, "
        #              f"g({coord_j:.4f})={g_j:.4f}")

        case = self.determine_case_for_strip(strip_interest)
        # logging.info(f"[find_optimal_division_point] Determined Case: {case}")

        if case == 1:
            return self.handle_case_1(coord_j_minus_1, coord_j, g_j_minus_1, g_j)
        elif case == 2:
            return self.handle_case_2(coord_j_minus_1, coord_j, g_j_minus_1, g_j)
        elif case == 3:
            return self.handle_case_3(coord_j_minus_1, coord_j, g_j_minus_1, g_j)

        # Fallback: midpoint
        return 0.5 * (coord_j_minus_1 + coord_j)

    def g(self, cut_coord):
        """
        g(cut_coord) = WCRT_left - WCRT_right.
        """
        return self.sp.calculate_wcrt_left(cut_coord) - self.sp.calculate_wcrt_right(cut_coord)

    def g_prime(self, cut_coord):
        """
        Numerical derivative for Newton-based methods.
        """
        h = 1e-6
        return (self.g(cut_coord + h) - self.g(cut_coord - h)) / (2 * h)

    # ---------------------------
    #   Finding the strip
    # ---------------------------
    def find_strip_of_interest(self):
        """
        Chooses the sub-interval (strip) in which the WCRT crossing likely occurs.
        Returns (coord_prev, coord_curr, strip_geom) or None.
        """
        best_strip = None
        min_wcrt_diff = float('inf')

        for coord_prev, coord_curr, strip_geometry in self.sp.strips:
            # Evaluate left vs. right WCRT at coord_curr
            d_left_j = self.sp.calculate_diagonal_at_coordinate(coord_curr)
            P_left_j = self.sp.query_accumulated_perimeter(coord_curr)
            WCRT_left_j = d_left_j + 0.5 * P_left_j

            P_right_j = self.P_R - P_left_j
            d_right_j = self.sp.calculate_diagonal_right(coord_curr)
            WCRT_right_j = d_right_j + 0.5 * P_right_j

            wcrt_diff = abs(WCRT_left_j - WCRT_right_j)
            if wcrt_diff < min_wcrt_diff:
                best_strip = (coord_prev, coord_curr, strip_geometry)
                min_wcrt_diff = wcrt_diff

            # If left > right => we found a crossing
            if WCRT_left_j > WCRT_right_j:
                # logging.info("[find_strip_of_interest] returning crossing boundary strip.")
                return coord_prev, coord_curr, strip_geometry

        # If never returned => use best strip if it exists
        # logging.info("[find_strip_of_interest] no crossing found; using best strip.")
        return best_strip

    # ---------------------------
    #   Determine case for strip
    # ---------------------------
    def determine_case_for_strip(self, strip):
        """
        Returns 1, 2, or 3.
          1 => no obstacle perimeter
          2 => obstacles present, no degeneracy
          3 => degeneracy (an obstacle edge collinear with boundary)
        """
        coord_prev, coord_curr, strip_geometry = strip
        obstacle_portions = self.get_obstacles_within_strip(strip_geometry)
        if not obstacle_portions:
            return 1

        eff_perim = self.sp.compute_strip_perimeter(strip_geometry)
        if eff_perim < 1e-9:
            return 1

        # Check for degeneracy
        for obs_geom in obstacle_portions:
            edges = self._extract_edges(obs_geom)
            for (p1, p2) in edges:
                x1, y1 = p1
                x2, y2 = p2
                if self.axis == 'x':
                    # vertical edge at x=coord_curr => degeneracy
                    if abs(x1 - x2) < 1e-9 and abs(x1 - coord_curr) < 1e-9:
                        return 3
                else:  # axis='y'
                    if abs(y1 - y2) < 1e-9 and abs(y1 - coord_curr) < 1e-9:
                        return 3

        return 2

    def get_obstacles_within_strip(self, strip_geometry):
        """Return obstacles intersecting 'strip_geometry'."""
        result = []
        for obs in self.sp.obstacles:
            if strip_geometry.intersects(obs):
                clipped = make_valid(strip_geometry.intersection(obs))
                if not clipped.is_empty:
                    result.append(clipped)
        return result

    def _extract_edges(self, geom):
        """
        Extract edges from geometry boundary => list[(p1, p2)].
        """
        edges = []
        if not geom or geom.is_empty:
            return edges

        boundary = geom.boundary
        if boundary is None or boundary.is_empty:
            return edges

        def _append_linestring_edges(ls):
            coords = list(ls.coords)
            for i in range(len(coords) - 1):
                edges.append((coords[i], coords[i+1]))

        if boundary.geom_type == 'LineString':
            _append_linestring_edges(boundary)
        elif boundary.geom_type == 'MultiLineString':
            for ls in boundary.geoms:
                if not ls.is_empty:
                    _append_linestring_edges(ls)
        elif boundary.geom_type == 'GeometryCollection':
            for subgeom in boundary.geoms:
                if subgeom.geom_type == 'LineString':
                    _append_linestring_edges(subgeom)
        return edges

    # ---------------------------
    #   Case Handling (1,2,3)
    # ---------------------------
    def handle_case_1(self, c_j_minus_1, c_j, g_j_minus_1, g_j):
        if g_j_minus_1 * g_j < 0:
            return self.apply_numerical_method(c_j_minus_1, c_j)
        return c_j

    def handle_case_2(self, c_j_minus_1, c_j, g_j_minus_1, g_j):
        if g_j_minus_1 * g_j < 0:
            return self.apply_numerical_method(c_j_minus_1, c_j)
        return c_j

    def handle_case_3(self, c_j_minus_1, c_j, g_j_minus_1, g_j):
        # Nudging boundary
        delta = 1e-6
        c_j_minus_delta = c_j - delta
        g_j_minus_delta = self.g(c_j_minus_delta)

        if g_j_minus_delta <= 0:
            return c_j
        if g_j_minus_1 * g_j_minus_delta < 0:
            return self.apply_numerical_method(c_j_minus_1, c_j_minus_delta)
        return c_j

    # ---------------------------
    #   Numerical Method
    # ---------------------------
    def apply_numerical_method(self, a, b):
        """
        Apply either 'brent' or 'newton' root-finding within [a,b].
        """
        # logging.info(f"[apply_numerical_method] bracket=[{a:.4f}, {b:.4f}]")
        bracket = [a, b]
        if self.method == 'brent':
            return solve_for_root_brent(self.g, a, b)
        else:
            x0 = 0.5 * (a + b)
            return solve_for_root_with_defensive_newton_rhapson(self.g, self.g_prime, x0, bracket=bracket)

    # ---------------------------
    #   Divide region
    # ---------------------------
    def divide_region(self, cut_coord):
        """
        Clips region & obstacles into left/right (axis='x') or top/bottom (axis='y').
        """
        if cut_coord is None or not isinstance(cut_coord, (int,float)):
            raise ValueError(f"[divide_region] Invalid cut_coord: {cut_coord}")

        # logging.info(f"[divide_region] dividing at {cut_coord:.4f}")
        minx, miny, maxx, maxy = self.sp.region.bounds

        # Build dividing polygons
        if self.axis == 'x':
            left_box = Polygon([(minx, miny), (cut_coord, miny),
                                (cut_coord, maxy), (minx, maxy)])
            right_box = Polygon([(cut_coord, miny), (maxx, miny),
                                 (maxx, maxy), (cut_coord, maxy)])
        else:
            left_box = Polygon([(minx, miny), (maxx, miny),
                                (maxx, cut_coord), (minx, cut_coord)])
            right_box = Polygon([(minx, cut_coord), (maxx, cut_coord),
                                 (maxx, maxy), (minx, maxy)])

        # Intersect
        R_left_raw = self.sp.region.intersection(left_box)
        R_right_raw = self.sp.region.intersection(right_box)
        R_left = make_valid(R_left_raw)
        R_right = make_valid(R_right_raw)

        if R_left.is_empty or R_right.is_empty:
            raise ValueError("[divide_region] One subregion is empty => invalid cut.")

        left_obstacles = []
        right_obstacles = []

        for obs in self.sp.obstacles:
            clipped_left = make_valid(obs.intersection(R_left))
            clipped_right = make_valid(obs.intersection(R_right))

            poly_left = self._extract_polygonal_part(clipped_left)
            poly_right = self._extract_polygonal_part(clipped_right)

            if poly_left is not None:
                left_obstacles.append(poly_left)
            if poly_right is not None:
                right_obstacles.append(poly_right)

        # logging.info(f"[divide_region] R_left area={R_left.area:.2f}, R_right area={R_right.area:.2f}")
        R_left_region, R_left_obstacles = validate_and_fix_geometries(R_left, left_obstacles)
        R_right_region, R_right_obstacles = validate_and_fix_geometries(R_right, right_obstacles)

        return (R_left_region, R_left_obstacles), (R_right_region, R_right_obstacles)

    def _extract_polygonal_part(self, geom):
        """
        Keep only Polygon/MultiPolygon from 'geom'; discard lines/points.
        """
        if geom.is_empty:
            return None

        gtype = geom.geom_type
        if gtype == "Polygon":
            return geom
        elif gtype == "MultiPolygon":
            return geom
        elif gtype in ("LineString", "MultiLineString", "Point", "MultiPoint"):
            return None
        elif gtype == "GeometryCollection":
            polygons = []
            for g in geom.geoms:
                if g.geom_type == "Polygon":
                    polygons.append(g)
                elif g.geom_type == "MultiPolygon":
                    polygons.extend(list(g.geoms))
            if not polygons:
                return None
            unified = unary_union(polygons)
            if unified.geom_type in ("Polygon","MultiPolygon"):
                return unified
            else:
                return None
        return None

def validate_and_fix_geometries(region, obstacles):
    """
    Validate/fix region + obstacle polygons.
    """
    if not region.is_valid:
        region = region.buffer(0)

    fixed_obs = []
    for obs in obstacles:
        if not obs.is_valid:
            obs = obs.buffer(0)
        fixed_obs.append(obs)

    return region, fixed_obs

