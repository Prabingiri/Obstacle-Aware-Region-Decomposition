from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
from shapely.ops import unary_union, transform
from shapely.validation import explain_validity
from shapely.errors import TopologicalError
import logging

logging.basicConfig(level=logging.INFO)


class RegionWithObstacles:
    """
    A class to preprocess and validate a region with obstacles for spatial analysis.

    Handles validation, merging, and clipping of obstacles within a region boundary.
    """

    def __init__(self, region_geom, obstacle_coords_list):
        """
        Initializes the RegionWithObstacles instance.

        Parameters:
        - region_geom (Polygon or MultiPolygon): The region boundary geometry.
        - obstacle_coords_list (list of list of tuple): List of coordinates for each obstacle.

        Raises:
        - ValueError: If the region geometry is invalid or disconnected.
        """
        self.region = self._validate_and_fix_region(region_geom)
        self.obstacles = self._create_and_clip_obstacles(obstacle_coords_list)
        self.merged_obstacles = self._merge_obstacles(self.obstacles)

    @staticmethod
    def drop_z_coordinates(geometry):
        """
        Drops the Z-coordinate from a geometry, ensuring it is 2D.

        Args:
            geometry (Geometry): Input geometry (Polygon, MultiPolygon, etc.)

        Returns:
            Geometry: Geometry with Z-coordinate removed.
        """
        if geometry.has_z:
            geometry = transform(lambda x, y, z=None: (x, y), geometry)
        return geometry

    def _validate_and_fix_region(self, region_geom):
        """
        Validates and ensures the region geometry is 2D and valid.

        Parameters:
        - region_geom (Polygon or MultiPolygon): The region boundary geometry.

        Returns:
        - region (Polygon or MultiPolygon): The validated region geometry.

        Raises:
        - ValueError: If the region geometry is invalid.
        """
        region_geom = self.drop_z_coordinates(region_geom)

        if not isinstance(region_geom, (Polygon, MultiPolygon)):
            raise ValueError("The region must be a Polygon or MultiPolygon.")

        if not region_geom.is_valid:
            logging.warning("Region geometry is invalid. Attempting to fix.")
            region_geom = region_geom.buffer(0)

        if not region_geom.is_valid:
            raise ValueError(f"Region geometry is invalid: {explain_validity(region_geom)}")

        return region_geom

    def _create_and_clip_obstacles(self, obstacle_coords_list):
        """
        Creates, validates, and clips obstacles to the region boundary.

        Parameters:
        - obstacle_coords_list (list of list of tuple): List of coordinates for each obstacle.

        Returns:
        - list[Polygon]: List of validated and clipped obstacle polygons.
        """
        obstacles = []
        for i, coords in enumerate(obstacle_coords_list, start=1):
            obstacle = self._create_and_validate_polygon(coords, f"Obstacle {i}")
            if obstacle is not None:
                obstacle = self.drop_z_coordinates(obstacle)  # Ensure obstacle is 2D
                clipped_obstacle = obstacle.intersection(self.region)
                if not clipped_obstacle.is_empty:
                    obstacles.append(clipped_obstacle)
                else:
                    logging.warning(f"Obstacle {i} lies entirely outside the region and was discarded.")
            else:
                logging.warning(f"Obstacle {i} is invalid and was discarded.")

        return obstacles

    def _merge_obstacles(self, obstacles):
        """
        Merges overlapping or adjacent obstacles into a single geometry.

        Parameters:
        - obstacles (list[Polygon or MultiPolygon]): List of obstacle polygons or multipolygons.

        Returns:
        - list[Polygon]: List of merged obstacles as individual Polygon objects.
        """
        if not obstacles:
            logging.warning("No obstacles to merge.")
            return []

        try:
            # Merge obstacles using unary_union
            merged = unary_union(obstacles)

            # Handle different types of results from unary_union
            if isinstance(merged, Polygon):
                return [merged]  # Single merged Polygon
            elif isinstance(merged, MultiPolygon):
                return list(merged.geoms)  # List of individual Polygons
            elif merged.is_empty:
                logging.warning("Merged obstacles result is empty.")
                return []
            else:
                # Unexpected result type from unary_union
                raise ValueError(f"Unexpected geometry type after merging: {type(merged)}")

        except TopologicalError as e:
            # Handle cases where merging fails due to invalid geometries
            logging.error(f"Error during obstacle merging: {e}")
            return obstacles  # Return input as-is to allow downstream handling

    def _create_and_validate_polygon(self, coords, name):
        """
        Creates and validates a polygon from coordinates.

        Parameters:
        - coords (list of tuple): Coordinates defining the polygon.
        - name (str): Name of the polygon for error messages.

        Returns:
        - polygon (Polygon): The validated polygon, or None if invalid.
        """
        polygon = Polygon(coords)
        validity = explain_validity(polygon)

        if validity != 'Valid Geometry':
            logging.warning(f"{name} is invalid ({validity}) and will be discarded.")
            return None

        return polygon

    def get_simplified_obstacles(self):
        """
        Returns the merged obstacles as a list of polygons.

        Returns:
        - list[Polygon]: Merged obstacles.
        """
        return self.merged_obstacles

    def check_region_connectivity(self):
        """
        Checks if the region is fully connected after subtracting obstacles.

        Returns:
        - bool: True if the region is connected, False otherwise.
        """
        remaining_region = self.region
        try:
            merged_obstacles = unary_union(self.merged_obstacles)
            remaining_region = remaining_region.difference(merged_obstacles)
        except TopologicalError as e:
            logging.error(f"Error during connectivity check: {e}")
            return False

        return remaining_region.geom_type == "Polygon"

    def create_region(self, region_geom):
        """
        Validates the region geometry.

        Parameters:
        - region_geom (Polygon or MultiPolygon): The region boundary geometry.

        Returns:
        - region (Polygon or MultiPolygon): The validated region geometry.

        Raises:
        - ValueError: If the region geometry is invalid.
        """
        region_geom = self.drop_z_coordinates(region_geom)  # Ensure region is 2D
        if not isinstance(region_geom, (Polygon, MultiPolygon)):
            raise ValueError("The region must be a Polygon or MultiPolygon.")

        if not region_geom.is_valid:
            logging.warning("Region geometry is invalid. Attempting to make it valid.")
            region_geom = region_geom.buffer(0)

        if not region_geom.is_valid:
            raise ValueError(f"The region geometry is invalid: {explain_validity(region_geom)}")

        return region_geom


    def visualize(self):
        """
        Visualizes the region, raw obstacles, and merged obstacles.
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        # Plot the region
        if isinstance(self.region, Polygon):
            x, y = self.region.exterior.xy
            ax.plot(x, y, color='black', label='Region Boundary')

        # Plot raw obstacles
        for obstacle in self.obstacles:
            x, y = obstacle.exterior.xy
            ax.fill(x, y, color='red', alpha=0.5, label='Obstacle')

        # Plot merged obstacles
        for geom in self.merged_obstacles:
            x, y = geom.exterior.xy
            ax.plot(x, y, color='blue', linestyle='--', label='Merged Obstacle')

        ax.set_title("Region with Obstacles")
        ax.legend()
        plt.show()
