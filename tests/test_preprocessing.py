import unittest
import logging
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.errors import TopologicalError
from shapely.ops import unary_union

from src.preprocessing import RegionWithObstacles

# Disable logging during tests to keep the output clean
logging.disable(logging.CRITICAL)


class TestRegionWithObstacles(unittest.TestCase):
    def setUp(self):
        # Create a simple square region (valid Polygon)
        self.region_polygon = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        # Define some obstacles:
        # 1. An obstacle completely inside the region.
        self.obstacle_inside = [(2, 2), (4, 2), (4, 4), (2, 4)]
        # 2. An obstacle completely outside the region.
        self.obstacle_outside = [(20, 20), (22, 20), (22, 22), (20, 22)]
        # 3. Two overlapping obstacles.
        self.obstacle_overlap_1 = [(3, 3), (6, 3), (6, 6), (3, 6)]
        self.obstacle_overlap_2 = [(5, 5), (8, 5), (8, 8), (5, 8)]
        # 4. An obstacle that divides the region.
        self.obstacle_dividing = [(5, 0), (6, 0), (6, 10), (5, 10)]

    def test_valid_region(self):
        """
        Test that a valid region polygon is accepted without modification.
        """
        rwo = RegionWithObstacles(self.region_polygon, [])
        # The region property should be valid and equal (geometrically) to the original.
        self.assertTrue(rwo.region.is_valid)
        self.assertAlmostEqual(rwo.region.area, self.region_polygon.area)

    def test_invalid_region_fix(self):
        """
        Test that an invalid (self-intersecting) region is fixed (via buffer(0)) if possible.
        """
        # Create a self-intersecting polygon (bow-tie shape)
        invalid_region = Polygon([(0, 0), (5, 5), (0, 5), (5, 0)])
        try:
            rwo = RegionWithObstacles(invalid_region, [])
        except ValueError:
            self.fail("RegionWithObstacles raised ValueError even though the region could be fixed.")
        else:
            self.assertTrue(rwo.region.is_valid)

    def test_obstacle_clipping(self):
        """
        Test that obstacles are clipped to the region: obstacles outside the region are discarded.
        """
        # Provide one obstacle inside and one completely outside the region.
        obstacles = [self.obstacle_inside, self.obstacle_outside]
        rwo = RegionWithObstacles(self.region_polygon, obstacles)
        # Check that the obstacle outside the region does not contribute to the merged obstacles.
        merged = rwo.get_simplified_obstacles()
        # Expect that the area of merged obstacles is equal to the area of the inside obstacle.
        expected_inside = Polygon(self.obstacle_inside)
        # If there is a merge, it might have produced one geometry.
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged[0].area, expected_inside.area)

    def test_merge_obstacles(self):
        """
        Test that overlapping obstacles are merged into one geometry.
        """
        obstacles = [self.obstacle_overlap_1, self.obstacle_overlap_2]
        rwo = RegionWithObstacles(self.region_polygon, obstacles)
        merged = rwo.get_simplified_obstacles()
        # Since the obstacles overlap, we expect a single merged polygon.
        self.assertEqual(len(merged), 1)
        # The merged area should be less than the sum of the individual areas but more than either one.
        area1 = Polygon(self.obstacle_overlap_1).area
        area2 = Polygon(self.obstacle_overlap_2).area
        merged_area = merged[0].area
        self.assertGreaterEqual(merged_area, max(area1, area2))
        self.assertLessEqual(merged_area, area1 + area2)

    def test_check_region_connectivity(self):
        """
        Test that the connectivity check identifies when the region is divided by an obstacle.
        """
        # An obstacle that spans from one side of the region to the other should disconnect it.
        obstacles = [self.obstacle_dividing]
        rwo = RegionWithObstacles(self.region_polygon, obstacles)
        connected = rwo.check_region_connectivity()
        self.assertFalse(connected)

    def test_drop_z_coordinates(self):
        """
        Test that geometries with Z coordinates are converted to 2D.
        """
        # Create a polygon with Z coordinates.
        coords_3d = [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
        polygon_3d = Polygon(coords_3d)
        polygon_2d = RegionWithObstacles.drop_z_coordinates(polygon_3d)
        # Check that the resulting coordinates have only two values.
        for coord in polygon_2d.exterior.coords:
            self.assertEqual(len(coord), 2)

    def test_create_and_validate_polygon(self):
        """
        Test the helper function _create_and_validate_polygon for both valid and invalid inputs.
        """
        rwo = RegionWithObstacles(self.region_polygon, [])
        valid_coords = [(0, 0), (4, 0), (4, 4), (0, 4)]
        polygon = rwo._create_and_validate_polygon(valid_coords, "Test Polygon")
        self.assertIsNotNone(polygon)
        # Create a self-intersecting polygon that should be deemed invalid.
        invalid_coords = [(0, 0), (4, 4), (4, 0), (0, 4)]
        polygon_invalid = rwo._create_and_validate_polygon(invalid_coords, "Test Polygon")
        self.assertIsNone(polygon_invalid)

    def test_create_region(self):
        """
        Test the public create_region method to ensure it validates and fixes the region geometry.
        """
        # Create a polygon with Z coordinates and a potential invalidity that can be fixed.
        coords_3d = [(0, 0, 5), (10, 0, 5), (10, 10, 5), (0, 10, 5)]
        region_3d = Polygon(coords_3d)
        rwo = RegionWithObstacles(self.region_polygon, [])
        fixed_region = rwo.create_region(region_3d)
        self.assertTrue(fixed_region.is_valid)
        # Check that the geometry is 2D (the coordinates should be tuples of length 2).
        for coord in fixed_region.exterior.coords:
            self.assertEqual(len(coord), 2)


if __name__ == '__main__':
    unittest.main()
