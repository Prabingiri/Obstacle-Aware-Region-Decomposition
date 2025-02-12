# src/module0.py

import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union, transform
from shapely.validation import make_valid
import pyproj


class DataPreprocessor:
    """
    A class to preprocess spatial data
    """

    def __init__(self, region_file_path, obstacles_file_path, target_epsg=32615):
        """
        Initialize the DataPreprocessor class.

        Args:
        - region_file_path (str): Path to the region GeoJSON file.
        - obstacles_file_path (str): Path to the obstacles GeoJSON file.
        - target_epsg (int): EPSG code for target CRS. Defaults to UTM Zone 15N (EPSG:32615).
        """
        self.region_file_path = region_file_path
        self.obstacles_file_path = obstacles_file_path
        self.target_epsg = target_epsg
        self.region = None
        self.obstacles = None

    def read_region(self):
        """
        Reads and preprocesses the region boundary from the GeoJSON file.

        Returns:
        - region_geometry (Polygon or MultiPolygon): The region boundary geometry.
        """
        gdf = gpd.read_file(self.region_file_path)
        gdf = self._ensure_crs_and_reproject(gdf)
        # Merge all geometries into a single geometry and validate
        geom = unary_union(gdf.geometry)
        geom = make_valid(self._drop_z_coordinates(geom))
        geom = geom.buffer(0)
        self.region = geom
        return self.region

    def read_obstacles(self):
        gdf = gpd.read_file(self.obstacles_file_path)
        gdf = self._ensure_crs_and_reproject(gdf)
        obstacles = []

        for geom in gdf.geometry:
            geom = make_valid(self._drop_z_coordinates(geom))

            # If it's a GeometryCollection, extract each sub-geom
            if geom.geom_type == "GeometryCollection":
                for subgeom in geom.geoms:
                    subgeom = make_valid(self._drop_z_coordinates(subgeom))
                    if isinstance(subgeom, (Polygon, MultiPolygon)):
                        obstacles.append(subgeom)

            # Otherwise, we keep polygons or multipolygons directly
            elif isinstance(geom, (Polygon, MultiPolygon)):
                obstacles.append(geom)
        # output_path = "FAA_obstacles.geojson"
        # Convert the list of obstacles into a GeoDataFrame
        # obstacles_gdf = gpd.GeoDataFrame(geometry=obstacles, crs=gdf.crs)

        # Write the GeoDataFrame to a GeoJSON file
        # obstacles_gdf.to_file(output_path, driver='GeoJSON')

        self.obstacles = obstacles
        return self.obstacles

    def get_obstacle_coordinates_list(self):
        """
        Gets the coordinates of the obstacles suitable for Module 1.

        Returns:
        - obstacle_coords_list (list of list of tuple): List of obstacle coordinates.
        """
        if self.obstacles is None:
            self.read_obstacles()
        obstacle_coords_list = []
        for obstacle in self.obstacles:
            if isinstance(obstacle, Polygon):
                coords = list(obstacle.exterior.coords)
                obstacle_coords_list.append(coords)
            elif isinstance(obstacle, MultiPolygon):
                for poly in obstacle.geoms:
                    coords = list(poly.exterior.coords)
                    obstacle_coords_list.append(coords)
        return obstacle_coords_list

    def preprocess(self):
        """
        Executes the full preprocessing pipeline.

        Returns:
        - region_geometry (Polygon or MultiPolygon): The region boundary geometry.
        - obstacle_coords_list (list of list of tuple): List of obstacle coordinates.
        """
        self.read_region()
        self.read_obstacles()
        obstacle_coords_list = self.get_obstacle_coordinates_list()
        return self.region, obstacle_coords_list

    def _ensure_crs_and_reproject(self, gdf):
        """
        Ensures CRS is set and reprojects the GeoDataFrame to the target CRS.

        Args:
        - gdf (GeoDataFrame): Input GeoDataFrame.

        Returns:
        - gdf (GeoDataFrame): Reprojected GeoDataFrame.
        """
        if gdf.crs is None:
            # Assuming EPSG:4269 as default if CRS is missing
            gdf.set_crs(epsg=4269, inplace=True)
        gdf = gdf.to_crs(epsg=self.target_epsg)
        return gdf

    @staticmethod
    def _drop_z_coordinates(geometry):
        """
        Drops the Z-coordinate from a geometry, ensuring it is 2D.

        Args:
        - geometry (Geometry): Input geometry (Polygon, MultiPolygon, etc.)

        Returns:
        - Geometry: Geometry with Z-coordinate removed.
        """
        if geometry.has_z:
            geometry = transform(lambda x, y, z=None: (x, y), geometry)
        return geometry
