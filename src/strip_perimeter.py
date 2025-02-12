import math
import logging
from shapely.geometry import Polygon, box, LineString, MultiPolygon, MultiLineString, GeometryCollection
from shapely.ops import unary_union, transform
from src.strip_visualization import visualize_strip, visualize_all_strips


class Strip:
    def __init__(self, region, obstacles, axis='x'):
        """
        Initialize the Strip class with a region, obstacles, and axis.

        Args:
            region (Polygon): The region polygon.
            obstacles (list[Polygon]): List of obstacle polygons.
            axis (str): The axis for calculations ('x' or 'y'). Defaults to 'x'.
        """
        if axis not in {'x', 'y'}:
            raise ValueError("Axis must be 'x' or 'y'.")

        self.region, self.obstacles = validate_and_fix_geometries(region, obstacles)

        # Ensure geometry is in 2D
        self.region = self._validate_2d(self.region)
        self.obstacles = [self._validate_2d(obs) for obs in self.obstacles]

        self.axis = axis
        self.events = self.define_events()
        self.strips = self.create_strips()

        # Dictionaries to hold perimeter data
        self.per_strip_perimeters = {}
        self.cumulative_perimeters = {}

        # Pre-compute the region boundary (could be MultiLineString)
        self._region_boundary = self._get_region_boundary()

        # Compute the perimeter contributions per strip
        self.compute_perimeters()

    @staticmethod
    def _validate_2d(geometry):
        """
        If the geometry has a Z-coordinate, drop it and return only the 2D geometry.
        """
        if geometry.has_z:
            geometry = transform(lambda x, y, z=None: (x, y), geometry)
        return geometry

    def _get_region_boundary(self):
        """
        Obtain a geometry (LineString or MultiLineString) that represents
        the outer boundary of self.region.
        If region is Polygon => single boundary.
        If region is MultiPolygon => possibly multiple boundaries.
        """
        if self.region.is_empty:
            logging.warning("Region geometry is empty.")
            return None

        if isinstance(self.region, Polygon):
            return LineString(self.region.exterior.coords)

        elif isinstance(self.region, MultiPolygon):
            # Gather the exterior of each sub-polygon into one MultiLineString
            boundary_parts = []
            for poly in self.region.geoms:
                if not poly.is_empty:
                    boundary_parts.append(LineString(poly.exterior.coords))
            if len(boundary_parts) == 1:
                return boundary_parts[0]  # a single LineString
            else:
                return unary_union(boundary_parts)  # typically a MultiLineString
        else:
            # If it's something else (GeometryCollection, etc.), fallback:
            return self.region.boundary


    def define_events(self):
        """
        Defines unique 'event' coordinates along the chosen axis (x or y).

        Returns:
            list[float]: Sorted list of unique event coordinates.
        """
        events = set()
        minx, miny, maxx, maxy = self.region.bounds

        if self.axis == 'x':
            # Region boundary
            events.update([minx, maxx])
            # Obstacle boundaries
            for obs in self.obstacles:
                for geom in (obs.geoms if isinstance(obs, MultiPolygon) else [obs]):
                    events.update([pt[0] for pt in geom.exterior.coords])
        else:  # axis == 'y'
            events.update([miny, maxy])
            for obs in self.obstacles:
                for geom in (obs.geoms if isinstance(obs, MultiPolygon) else [obs]):
                    events.update([pt[1] for pt in geom.exterior.coords])

        return sorted(events)


    def create_strips(self):
        """
        Creates strip geometries between consecutive unique events along the chosen axis.

        Returns:
            list[tuple]: Each tuple is (start_coord, end_coord, strip_polygon).
        """
        strips = []
        for i in range(1, len(self.events)):
            c_prev = self.events[i - 1]
            c_curr = self.events[i]

            # Build a strip from c_prev to c_curr
            minx, miny, maxx, maxy = self.region.bounds
            if self.axis == 'x':
                strip_poly = box(c_prev, miny, c_curr, maxy)
            else:  # axis == 'y'
                strip_poly = box(minx, c_prev, maxx, c_curr)

            strips.append((c_prev, c_curr, strip_poly))

        return strips

    def compute_perimeters(self):
        """
        Computes the perimeter contributed by each strip and stores both
        per-strip perimeter and the cumulative perimeter.
        """
        accumulated_perimeter = 0.0
        for (coord_prev, coord_curr, strip) in self.strips:
            perimeter = self.compute_strip_perimeter(strip)
            accumulated_perimeter += perimeter

            # Store these results
            self.per_strip_perimeters[(coord_prev, coord_curr)] = perimeter
            self.cumulative_perimeters[coord_curr] = accumulated_perimeter



    def compute_strip_perimeter(self, strip):
        total_perimeter = 0.0
        coord_prev = strip.bounds[0] if self.axis == 'x' else strip.bounds[1]

        # Intersect strip with region
        strip_in_region = strip.intersection(self.region)
        if strip_in_region.is_empty:
            return 0.0

        region_boundary = self._get_region_boundary()
        if not region_boundary or region_boundary.is_empty:
            logging.warning("Region boundary is empty or None. Returning 0 perimeter.")
            return 0.0

        for obs in self.obstacles:
            if isinstance(obs, Polygon):
                boundaries = [obs.exterior]
            elif isinstance(obs, MultiPolygon):
                boundaries = [poly.exterior for poly in obs.geoms]
            else:
                logging.warning(f"Unexpected obstacle type: {type(obs)}")
                continue

            for boundary in boundaries:
                coords_list = list(boundary.coords)
                for i in range(len(coords_list) - 1):
                    edge = LineString([coords_list[i], coords_list[i + 1]])
                    clipped_edge = edge.intersection(strip_in_region)  # <--- intersection with strip_in_region

                    if clipped_edge.is_empty:
                        continue

                    # Flatten region boundary overlap
                    unaligned_edge = self._exclude_aligned_portions(clipped_edge, region_boundary)

                    if unaligned_edge:
                        # unaligned_edge can be LineString or MultiLineString
                        if isinstance(unaligned_edge, LineString):
                            # skip if fully collinear with lower boundary
                            if not self.is_edge_collinear_with_coord(unaligned_edge, coord_prev, self.axis):
                                total_perimeter += unaligned_edge.length

                        elif isinstance(unaligned_edge, MultiLineString):
                            # sum each line if it's not collinear
                            for sub_line in unaligned_edge.geoms:
                                if not self.is_edge_collinear_with_coord(sub_line, coord_prev, self.axis):
                                    total_perimeter += sub_line.length

        return total_perimeter


    def _exclude_aligned_portions(self, edge, region_boundary):
        """
        Excludes the portion of 'edge' that aligns with the region boundary.
        Returns a flattened collection of line(s) or None.
        """
        aligned_portion = edge.intersection(region_boundary)

        if aligned_portion.is_empty:
            # No overlap => entire edge is valid
            return edge

        # Subtract any portion that lies exactly on the boundary
        unaligned_portion = edge.difference(aligned_portion)
        if unaligned_portion.is_empty:
            return None

        return self._flatten_to_lines(unaligned_portion)

    def _flatten_to_lines(self, geom):
        """
        Convert 'geom' (which may be a LineString, MultiLineString, GeometryCollection, etc.)
        into a MultiLineString or list of LineStrings. Return None if no line segments.
        """
        from shapely.geometry import (
            LineString, MultiLineString, GeometryCollection
        )

        if geom.is_empty:
            return None

        if isinstance(geom, LineString):
            return geom  # single line is fine

        if isinstance(geom, MultiLineString):
            # Return directly, or unify them if you wish
            return geom

        if isinstance(geom, GeometryCollection):
            lines = []
            for g in geom.geoms:
                if g.is_empty:
                    continue
                if isinstance(g, LineString):
                    lines.append(g)
                elif isinstance(g, MultiLineString):
                    lines.extend(list(g.geoms))
                # ignore Points or Polygons, as we only measure line perimeter
            if not lines:
                return None
            if len(lines) == 1:
                return lines[0]
            return MultiLineString(lines)

        # fallback
        return None


    @staticmethod
    def is_edge_collinear_with_coord(edge, coord, axis, eps=1e-9):
        """
        Checks if a line is entirely on x=coord (if axis='x') or y=coord (if axis='y'),
        within some small floating-point tolerance eps.
        """
        (x1, y1) = edge.coords[0]
        (x2, y2) = edge.coords[1]

        if axis == 'x':
            return (abs(x1 - coord) < eps) and (abs(x2 - coord) < eps)
        else:  # axis='y'
            return (abs(y1 - coord) < eps) and (abs(y2 - coord) < eps)

    def query_accumulated_perimeter(self, coord):
        """
        Queries the cumulative perimeter up to a given coordinate.

        Steps:
        1. If coord matches an existing event, return that precomputed value.
        2. Otherwise, get the cumulative perimeter at the last known event < coord.
        3. Add perimeter from that event up to 'coord' by creating a partial strip.
        """
        # 1. If exact event:
        if coord in self.cumulative_perimeters:
            return self.cumulative_perimeters[coord]

        # 2. Find last_event < coord
        last_event = None
        for event in sorted(self.cumulative_perimeters.keys()):
            if coord < event:
                break
            last_event = event

        cumulative_perimeter_query = self.cumulative_perimeters.get(last_event, 0.0)

        # 3. Create partial strip from last_event to coord
        if last_event is not None:
            for (coord_prev, coord_curr, _) in self.strips:
                if last_event == coord_prev and coord < coord_curr:
                    partial_strip = self._create_partial_strip(coord_prev, coord)
                    partial_perimeter = self.compute_strip_perimeter(partial_strip)
                    cumulative_perimeter_query += partial_perimeter
                    break

        return cumulative_perimeter_query

    def _create_partial_strip(self, coord_start, coord_end):
        """
        Creates a "partial" strip from coord_start to coord_end within region bounds.

        Args:
            coord_start (float): Starting coordinate.
            coord_end (float): Ending coordinate.

        Returns:
            Polygon: The partial strip geometry.
        """
        minx, miny, maxx, maxy = self.region.bounds

        if self.axis == 'x':
            return box(coord_start, miny, coord_end, maxy)
        else:  # axis == 'y'
            return box(minx, coord_start, maxx, coord_end)

    def query_custom_strip_perimeter(self, coord1, coord2, include_cumulative=False):
        """
        Calculate perimeter for a custom strip (coord1 -> coord2). Optionally include
        the cumulative perimeter up to min(coord1, coord2).

        Args:
            coord1 (float): One boundary of the strip.
            coord2 (float): Other boundary of the strip.
            include_cumulative (bool): If True, add the perimeter up to min(coord1, coord2).

        Returns:
            float: The perimeter of the custom strip, plus the optional cumulative portion.
        """
        # Ensure ordering
        if coord1 > coord2:
            coord1, coord2 = coord2, coord1

        # Check region bounds
        minx, miny, maxx, maxy = self.region.bounds
        if self.axis == 'x':
            if not (minx <= coord1 <= maxx and minx <= coord2 <= maxx):
                raise ValueError("Coordinates are outside the region's x-bounds.")
        else:
            if not (miny <= coord1 <= maxy and miny <= coord2 <= maxy):
                raise ValueError("Coordinates are outside the region's y-bounds.")

        # Build custom strip
        custom_strip = self._create_partial_strip(coord1, coord2)
        custom_strip_perimeter = self.compute_strip_perimeter(custom_strip)

        if include_cumulative:
            # Add perimeter up to coord1
            cumulative_perimeter_before = self.query_accumulated_perimeter(coord1)
            return cumulative_perimeter_before + custom_strip_perimeter

        return custom_strip_perimeter

    def calculate_total_obstacle_perimeter(self):
        """
        Calculates total perimeter of all obstacles, excluding portions aligning with region boundary.

        Returns:
            float: The total obstacle perimeter.
        """
        # region_boundary = LineString(self.region.exterior.coords)
        region_boundary = self._get_region_boundary()
        if not region_boundary or region_boundary.is_empty:
            logging.warning("Region boundary is empty or None. No obstacle perimeter subtractions.")
            # In such a scenario, either return sum of all obstacle exteriors
            return sum(obs.length for obs in self.obstacles if isinstance(obs, Polygon)) \
                   + sum(poly.length for mp in [o for o in self.obstacles if isinstance(o, MultiPolygon)]
                         for poly in mp.geoms)

        total_perimeter = 0.0

        for obs in self.obstacles:
            # Handle Polygon vs MultiPolygon
            if isinstance(obs, Polygon):
                boundaries = [obs.exterior]
            elif isinstance(obs, MultiPolygon):
                boundaries = [poly.exterior for poly in obs.geoms]
            else:
                logging.warning(f"Unexpected obstacle type: {type(obs)}")
                continue

            for boundary in boundaries:
                coords_list = list(boundary.coords)
                for i in range(len(coords_list) - 1):
                    edge = LineString([coords_list[i], coords_list[i + 1]])
                    unaligned_edge = self._exclude_aligned_portions(edge, region_boundary)
                    if unaligned_edge:
                        total_perimeter += unaligned_edge.length

        return total_perimeter

    def calculate_region_diagonal(self):
        """
        Calculates the full diagonal of the region's bounding box.

        Returns:
            float: Diagonal distance of the region.
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny
        return math.sqrt(width**2 + height**2)

    def calculate_region_wcrt(self):
        """
        Calculates the region-level WCRT:

           WCRT = Region Diagonal + 0.5 * Total Obstacle Perimeter
        """
        diag = self.calculate_region_diagonal()
        obstacle_perim = self.calculate_total_obstacle_perimeter()
        return diag + 0.5 * obstacle_perim

    def calculate_region_diagonal_half(self):
        """
        Calculates a half-diagonal measure depending on the axis.

        For axis='x': sqrt( H^2 + (W/2)^2 )
        For axis='y': sqrt( W^2 + (H/2)^2 )
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny

        if self.axis == 'x':
            return math.sqrt(height**2 + (width / 2)**2)
        else:  # axis == 'y'
            return math.sqrt(width**2 + (height / 2)**2)

    def calculate_target_wcrt(self):
        """
        Calculates a target WCRT: Half the diagonal plus 0.25 * total obstacle perimeter.
        """
        half_diag = self.calculate_region_diagonal_half()
        total_perim = self.calculate_total_obstacle_perimeter()
        return half_diag + 0.25 * total_perim

    def calculate_target_wcrt_dynamic(self):
        """
        Dynamically calculates a WCRT target based on the aspect ratio of the region.

        Weighted combination of an "effective" diagonal and 0.25 of the total obstacle perimeter.
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny

        # Effective diagonal at half-cut
        if self.axis == 'x':
            effective_diagonal = math.sqrt(height**2 + (width / 2)**2)
        else:  # 'y'
            effective_diagonal = math.sqrt(width**2 + (height / 2)**2)

        # Compute aspect ratio and weighting
        aspect_ratio = width / height if height != 0 else 1.0
        if self.axis == 'x':
            alpha = 1 / (1 + aspect_ratio)
            beta = aspect_ratio / (1 + aspect_ratio)
        else:  # 'y'
            alpha = aspect_ratio / (1 + aspect_ratio)
            beta = 1 / (1 + aspect_ratio)

        wcrt_target = alpha * effective_diagonal + beta * 0.25 * self.calculate_total_obstacle_perimeter()
        return wcrt_target

    def calculate_diagonal_at_coordinate(self, coord):
        """
        Calculates diagonal distance from the "lower-left" corner of the region
        to x=coord (if axis='x') or y=coord (if axis='y').
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny

        if self.axis == 'x':
            return math.sqrt(height**2 + (coord - minx)**2)
        else:  # axis == 'y'
            return math.sqrt(width**2 + (coord - miny)**2)

    def calculate_wcrt_at_strip(self, strip):
        """
        Calculates WCRT at a given strip by taking:
          WCRT(strip) = diagonal_distance_at_coord + 0.5 * cumulative_perimeter(coord)

        where coord = x-strip's 'upper' boundary (if axis='x'), or
              coord = y-strip's 'upper' boundary (if axis='y').
        """
        if self.axis == 'x':
            coord = strip.bounds[2]  # x_max of this strip
        else:
            coord = strip.bounds[3]  # y_max of this strip

        diag_at_coord = self.calculate_diagonal_at_coordinate(coord)
        cum_perim = self.cumulative_perimeters.get(coord, 0.0)
        return diag_at_coord + 0.5 * cum_perim

    def query_wcrt_at_coordinate(self, coord):
        """
        Calculates WCRT at an arbitrary coordinate.

        WCRT(coord) = diagonal_at_coord + 0.5 * perimeter_up_to(coord)
        """
        diag = self.calculate_diagonal_at_coordinate(coord)
        cum_perim = self.query_accumulated_perimeter(coord)
        return diag + 0.5 * cum_perim

    def calculate_wcrt_left(self, division_point):
        """
        Calculates the WCRT for the 'left' (if axis='x') or 'bottom' (if axis='y') subregion.

        WCRT_left = diagonal_left + 0.5 * perimeter_left
        """
        diag_left = self.calculate_diagonal_at_coordinate(division_point)
        perim_left = self.query_accumulated_perimeter(division_point)
        return diag_left + 0.5 * perim_left

    def calculate_wcrt_right(self, division_point):
        """
        Calculates WCRT for the 'right' (if axis='x') or 'top' (if axis='y') subregion.

        WCRT_right(division_point) = diagonal_right + 0.5 * (P_total - P_left)

        If axis='x':
           diagonal_right = sqrt(height^2 + (maxx - division_point)^2)
        If axis='y':
           diagonal_right = sqrt(width^2 + (maxy - division_point)^2)
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny

        if self.axis == 'x':
            diag_right = math.sqrt(height**2 + (maxx - division_point)**2)
        else:  # axis == 'y'
            diag_right = math.sqrt(width**2 + (maxy - division_point)**2)

        P_total = self.calculate_total_obstacle_perimeter()
        P_left = self.query_accumulated_perimeter(division_point)
        P_right = P_total - P_left

        return diag_right + 0.5 * P_right

    def calculate_diagonal_right(self, division_point):
        """
        Returns just the diagonal portion for the 'right' (axis='x') or 'top' (axis='y') subregion.
        Useful for debugging.

        axis='x': sqrt(height^2 + (maxx - division_point)^2)
        axis='y': sqrt(width^2 + (maxy - division_point)^2)
        """
        minx, miny, maxx, maxy = self.region.bounds
        width = maxx - minx
        height = maxy - miny

        if self.axis == 'x':
            return math.sqrt(height**2 + (maxx - division_point)**2)
        else:  # axis == 'y'
            return math.sqrt(width**2 + (maxy - division_point)**2)

    def visualize_all(self):
        """
        Visualizes all strips if you have 'strip_visualization.py' and the necessary plotting backends.
        """
        visualize_all_strips(
            self.region,
            self.obstacles,
            self.strips,
            self.per_strip_perimeters,
            self.cumulative_perimeters
        )

    def calculate_total_obstacle_area(self):
        """
        Computes the total area occupied by obstacles (unions them if needed).
        """
        total_area = 0.0
        for obs in self.obstacles:
            if obs.is_valid:
                total_area += obs.area
        return total_area

def validate_and_fix_geometries(region, obstacles):
        """
        Validates and fixes geometries for the region and obstacles.

        Args:
            region (Polygon): The region polygon.
            obstacles (list[Polygon]): List of obstacle polygons.

        Returns:
            tuple: Fixed region polygon and list of fixed obstacle polygons.
        """
        # Fix region if invalid
        if not region.is_valid:
            region = region.buffer(0)

        # Fix obstacles if invalid
        fixed_obstacles = []
        for obs in obstacles:
            if not obs.is_valid:
                obs = obs.buffer(0)
            fixed_obstacles.append(obs)

        return region, fixed_obstacles



#=================test=================
if __name__ == "__main__":
    # Example usage:
    region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
    hexagon = Polygon([(10, 20), (20, 10), (30, 20), (30, 30), (20, 40), (10, 30), (10, 20)])
    pentagon = Polygon([(40, 50), (50, 40), (60, 50), (55, 65), (45, 65), (40, 50)])
    triangle = Polygon([(70, 70), (90, 70), (80, 90), (70, 70)])
    rectangle1 = Polygon([(50, 20), (70, 20), (70, 40), (50, 40), (50, 20)])
    rectangle2 = Polygon([(60, 30), (80, 30), (80, 50), (60, 50), (60, 30)])

    obstacles = [hexagon, pentagon, triangle, rectangle1, rectangle2]

    # Instantiate Strip manager
    strip_manager = Strip(region, obstacles, axis='x')

    # 1. Region-wide WCRT
    wcrt_region = strip_manager.calculate_region_wcrt()
    print(f"Region WCRT: {wcrt_region:.2f}")

    # 2. WCRT at a specific coordinate
    wcrt_coord = strip_manager.query_wcrt_at_coordinate(50)
    print(f"WCRT at x=50: {wcrt_coord:.2f}")

    # 3. Split at x=40 and compute left vs. right subregion WCRT
    division_point = 40
    wcrt_left = strip_manager.calculate_wcrt_left(division_point)
    wcrt_right = strip_manager.calculate_wcrt_right(division_point)
    print(f"WCRT (x <= 40): {wcrt_left:.2f}")
    print(f"WCRT (x >= 40): {wcrt_right:.2f}")

    # Visualize strips (requires plotting support)
    strip_manager.visualize_all()
