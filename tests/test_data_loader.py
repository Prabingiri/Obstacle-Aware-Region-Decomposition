import unittest
import os
import tempfile
import geopandas as gpd
from shapely.geometry import Polygon, GeometryCollection
from shapely.ops import unary_union

from src.data_Loader import DataPreprocessor


class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        """
        Create temporary GeoJSON files for region and obstacles.
        Note: We use coordinates appropriate for EPSG:32615.
        """
        # Create a temporary directory for test files.
        self.temp_dir = tempfile.TemporaryDirectory()
        self.region_file_path = os.path.join(self.temp_dir.name, 'region.geojson')
        self.obstacles_file_path = os.path.join(self.temp_dir.name, 'obstacles.geojson')

        # Create a sample region in EPSG:4326 that covers an area in US
        # For example, a square around (–93, 45):
        region_geom = Polygon([
            (-93.5, 44.5),
            (-92.5, 44.5),
            (-92.5, 45.5),
            (-93.5, 45.5),
            (-93.5, 44.5)
        ])
        region_gdf = gpd.GeoDataFrame({'geometry': [region_geom]}, crs="EPSG:4326")
        region_gdf.to_file(self.region_file_path, driver='GeoJSON')

        # Create sample obstacles within the region.
        # Obstacle 1: A simple polygon.
        obstacle1 = Polygon([
            (-93.4, 44.6),
            (-93.3, 44.6),
            (-93.3, 44.7),
            (-93.4, 44.7),
            (-93.4, 44.6)
        ])
        # Obstacle 2: A GeometryCollection containing one polygon.
        obstacle2_poly = Polygon([
            (-93.2, 45.0),
            (-93.1, 45.0),
            (-93.1, 45.1),
            (-93.2, 45.1),
            (-93.2, 45.0)
        ])
        obstacle2 = GeometryCollection([obstacle2_poly])
        obstacles_gdf = gpd.GeoDataFrame({'geometry': [obstacle1, obstacle2]}, crs="EPSG:4326")
        obstacles_gdf.to_file(self.obstacles_file_path, driver='GeoJSON')

    def tearDown(self):
        """Clean up the temporary directory."""
        self.temp_dir.cleanup()

    def test_read_region(self):
        """
        Test that the region file is read, reprojected, and preprocessed correctly.
        """
        dp = DataPreprocessor(self.region_file_path, self.obstacles_file_path, target_epsg=32615)
        region = dp.read_region()
        # Verify that the region geometry is a Polygon or MultiPolygon and is valid.
        self.assertIn(region.geom_type, ["Polygon", "MultiPolygon"])
        self.assertTrue(region.is_valid)

    def test_read_obstacles(self):
        """
        Test that the obstacles file is read and that geometries are extracted properly.
        """
        dp = DataPreprocessor(self.region_file_path, self.obstacles_file_path, target_epsg=32615)
        obstacles = dp.read_obstacles()
        # We expect two obstacles to be extracted.
        self.assertEqual(len(obstacles), 2)
        for obs in obstacles:
            self.assertIn(obs.geom_type, ["Polygon", "MultiPolygon"])
            self.assertTrue(obs.is_valid)

    def test_get_obstacle_coordinates_list(self):
        """
        Test that the obstacle coordinates are correctly extracted from the obstacles.
        """
        dp = DataPreprocessor(self.region_file_path, self.obstacles_file_path, target_epsg=32615)
        dp.read_obstacles()
        coords_list = dp.get_obstacle_coordinates_list()
        # We expect two sets of coordinates.
        self.assertEqual(len(coords_list), 2)
        for coords in coords_list:
            self.assertIsInstance(coords, list)
            # Each coordinate should be a 2-tuple (x, y)
            for coord in coords:
                self.assertEqual(len(coord), 2)

    def test_preprocess(self):
        """
        Test the full preprocessing pipeline.
        """
        dp = DataPreprocessor(self.region_file_path, self.obstacles_file_path, target_epsg=32615)
        region, obstacle_coords_list = dp.preprocess()
        self.assertIn(region.geom_type, ["Polygon", "MultiPolygon"])
        self.assertIsInstance(obstacle_coords_list, list)
        self.assertEqual(len(obstacle_coords_list), 2)

    def test_ensure_crs_and_reproject(self):
        """
        Test that a GeoDataFrame with no CRS is assigned a default CRS and reprojected to the target.
        """
        poly = Polygon([(-93.5, 44.5), (-93.4, 44.5), (-93.4, 44.6), (-93.5, 44.6), (-93.5, 44.5)])
        gdf = gpd.GeoDataFrame({'geometry': [poly]})
        dp = DataPreprocessor(self.region_file_path, self.obstacles_file_path, target_epsg=32615)
        gdf_reproj = dp._ensure_crs_and_reproject(gdf)
        self.assertEqual(gdf_reproj.crs.to_epsg(), 32615)

    def test_drop_z_coordinates(self):
        """
        Test that a geometry with Z-coordinates is converted to 2D.
        """
        poly_3d = Polygon([
            (-93.5, 44.5, 100),
            (-93.4, 44.5, 100),
            (-93.4, 44.6, 100),
            (-93.5, 44.6, 100),
            (-93.5, 44.5, 100)
        ])
        poly_2d = DataPreprocessor._drop_z_coordinates(poly_3d)
        for coord in poly_2d.exterior.coords:
            self.assertEqual(len(coord), 2)


if __name__ == '__main__':
    unittest.main()
