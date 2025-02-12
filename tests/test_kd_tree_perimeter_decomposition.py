import math
import unittest
from shapely.geometry import Polygon, box
from src.kd_tree_perimeter_decomposition import KDTreePartitioning, validate_geometry


class TestKDTreePartitioning(unittest.TestCase):
    def setUp(self):
        # Create a simple 100x100 region.
        self.region = Polygon([
            (0, 0), (100, 0), (100, 100), (0, 100), (0, 0)
        ])
        # Create one obstacle: a 20x20 box located in the left half.
        self.obstacles = [box(20, 20, 40, 40)]

    def test_run_basic(self):
        """
        With max_depth=2 and advanced_checks disabled, expect the KD-Tree partitioner
        to produce at least one partition. Each partition should be a tuple (region, obstacles)
        with a valid, non-empty region.
        """
        kd = KDTreePartitioning(self.region, self.obstacles, max_depth=2, advanced_checks=False)
        partitions = kd.run()
        self.assertGreater(len(partitions), 0, "Expected at least one partition.")
        for region_part, obs_part in partitions:
            self.assertIsNotNone(region_part)
            self.assertFalse(region_part.is_empty)
            self.assertTrue(region_part.is_valid)

    def test_max_depth(self):
        """
        When max_depth is 0, no subdivision should occur and the partition list
        should contain exactly one partition identical (in area) to the original region.
        """
        kd = KDTreePartitioning(self.region, self.obstacles, max_depth=0)
        partitions = kd.run()
        self.assertEqual(len(partitions), 1, "At max_depth=0, expect exactly one partition.")
        region_part, obs_part = partitions[0]
        self.assertAlmostEqual(region_part.area, self.region.area,
                               msg="Partition area should equal original region area.")

    def test_advanced_checks_stop(self):
        """
        With advanced_checks enabled, if the obstacle coverage is very high,
        partitioning should stop and store the parent region.
        Here we simulate this by using a small region that is almost entirely covered.
        """
        # Create a small 10x10 region.
        region_small = Polygon([
            (0, 0), (10, 0), (10, 10), (0, 10), (0, 0)
        ])
        # Create an obstacle that nearly covers the region.
        obstacle_full = Polygon([
            (0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9), (0.1, 0.1)
        ])
        kd = KDTreePartitioning(region_small, [obstacle_full], max_depth=3, advanced_checks=True)
        partitions = kd.run()
        self.assertEqual(len(partitions), 1,
                         "Expected one partition when advanced checks stop splitting due to high coverage.")

    def test_partition_structure(self):
        """
        Verify that each partition is a tuple (region, obstacles), where region
        is a valid geometry (with an 'area' attribute) and obstacles is a list.
        """
        kd = KDTreePartitioning(self.region, self.obstacles, max_depth=2)
        partitions = kd.run()
        for part in partitions:
            self.assertIsInstance(part, tuple)
            self.assertEqual(len(part), 2)
            region_part, obs_part = part
            self.assertTrue(hasattr(region_part, "area"), "Region partition should have an 'area' attribute.")
            self.assertIsInstance(obs_part, list)


if __name__ == '__main__':
    unittest.main()
