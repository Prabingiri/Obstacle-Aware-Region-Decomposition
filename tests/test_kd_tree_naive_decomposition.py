import math
import unittest
from shapely.geometry import Polygon, box
from src.kd_tree_naive_decomposition import NaiveKDTreePartitioning, validate_and_fix_geometries, validate_geometry


class TestNaiveKDTreePartitioning(unittest.TestCase):
    def setUp(self):
        # Create a simple region: a 100x100 square.
        self.region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        # Create one obstacle: a 20x20 box located in the left half.
        self.obstacles = [box(20, 20, 40, 40)]

    def test_run_basic(self):
        """
        Test basic partitioning: With max_depth=2 and no advanced checks,
        we expect at least one partition with a valid region.
        """
        kd = NaiveKDTreePartitioning(self.region, self.obstacles, max_depth=2, advanced_checks=False)
        partitions = kd.run()
        self.assertGreater(len(partitions), 0, "Expected at least one partition.")
        for region_part, obs_part in partitions:
            # Check that each partition's region is valid and non-empty.
            self.assertIsNotNone(region_part)
            self.assertFalse(region_part.is_empty)
            self.assertTrue(region_part.is_valid)

    def test_stop_on_small_region(self):
        """
        If the region is very small (area below the min_area_threshold),
        the partitioning should stop and store the parent region.
        """
        small_region = box(0, 0, 0.001, 0.001)  # area ~1e-6
        kd = NaiveKDTreePartitioning(small_region, self.obstacles, max_depth=2, min_area_threshold=1e-3)
        partitions = kd.run()
        self.assertEqual(len(partitions), 1, "Expected one partition for a tiny region.")
        region_part, obs_part = partitions[0]
        # The stored region should equal the original small region.
        self.assertEqual(region_part, small_region)

    def test_max_depth(self):
        """
        When max_depth is set to 0, no recursive partitioning should occur;
        run() should return one partition that is the top-level region.
        """
        kd = NaiveKDTreePartitioning(self.region, self.obstacles, max_depth=0)
        partitions = kd.run()
        self.assertEqual(len(partitions), 1, "Expected one partition at max_depth=0.")
        region_part, obs_part = partitions[0]
        self.assertAlmostEqual(region_part.area, self.region.area, msg="Partition area should equal region area.")

    def test_advanced_checks(self):
        """
        With advanced_checks enabled, if the obstacle coverage is very high
        (leaving little free space), partitioning should stop and store the region.
        """
        # Create a small region (10x10) and an obstacle that nearly covers it.
        region = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        # Obstacle covers almost the entire region.
        obstacle = Polygon([(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9), (0.1, 0.1)])
        kd = NaiveKDTreePartitioning(region, [obstacle], max_depth=3, advanced_checks=True)
        partitions = kd.run()
        # Expect that advanced checks cause the partitioning to stop, storing the parent region.
        self.assertEqual(len(partitions), 1, "Expected one partition when advanced checks stop splitting.")

    def test_partition_structure(self):
        """
        Verify that each partition is a tuple (region, obstacles), where region is a valid geometry
        and obstacles is a list.
        """
        kd = NaiveKDTreePartitioning(self.region, self.obstacles, max_depth=2)
        partitions = kd.run()
        for part in partitions:
            self.assertIsInstance(part, tuple)
            self.assertEqual(len(part), 2)
            region_part, obs_part = part
            self.assertTrue(hasattr(region_part, "area"), "Region partition should have an area attribute.")
            self.assertIsInstance(obs_part, list)


if __name__ == '__main__':
    unittest.main()
