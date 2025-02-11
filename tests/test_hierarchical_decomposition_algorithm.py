import math
import unittest
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from src.hierarchical_decomposition_algorithm import HierarchicalDecomposition

# Constants defined in the module (or re-define them here if not imported)
DRONE_THRESHOLD = 5.0
COVERAGE_RATIO_STOP = 0.90


class TestHierarchicalDecomposition(unittest.TestCase):
    def setUp(self):
        # Default region: 100x100 square.
        self.region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        # Default obstacle: a 20x20 box, which leaves most of the region free.
        self.obstacles = [box(20, 20, 40, 40)]

    def test_validate_geometry(self):
        """Test that an invalid (self-intersecting) polygon is fixed."""
        # Create a bow-tie (self-intersecting polygon).
        invalid_poly = Polygon([(0, 0), (5, 5), (0, 5), (5, 0)])
        hd = HierarchicalDecomposition(invalid_poly, self.obstacles)
        valid_poly = hd._validate_geometry(invalid_poly)
        self.assertTrue(valid_poly.is_valid, "Geometry was not corrected to a valid one.")

    def test_is_subregion_valid_with_high_coverage(self):
        """
        Use a small region with an obstacle that nearly covers it.
        For example, a 10x10 region with an obstacle that covers almost the entire area.
        This should cause _is_subregion_valid to return False.
        """
        region = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        # Create an obstacle that almost covers the region.
        # For instance, an obstacle from (0.1,0.1) to (9.9,9.9) has area ~96.04.
        obstacle = Polygon([(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9), (0.1, 0.1)])
        hd = HierarchicalDecomposition(region, [obstacle], max_depth=3)
        self.assertFalse(hd._is_subregion_valid(region, [obstacle]),
                         "Subregion with high obstacle coverage should be invalid.")

    def test_decompose_max_depth(self):
        """
        When max_depth is 0, _decompose should store the top-level region immediately.
        """
        hd = HierarchicalDecomposition(self.region, self.obstacles, max_depth=0)
        partitions = hd.run()
        self.assertEqual(len(partitions), 1, "Max-depth=0 should yield exactly one partition.")
        self.assertAlmostEqual(partitions[0][0].area, self.region.area,
                               msg="The stored partition should be the entire region.")

    def test_run_with_valid_partitions(self):
        """
        Test that run() produces at least one partition for a region
        with modest obstacles.
        """
        hd = HierarchicalDecomposition(self.region, self.obstacles, max_depth=3)
        partitions = hd.run()
        self.assertGreater(len(partitions), 0, "Expected at least one partition to be produced.")

    def test_compute_largest_free_space(self):
        """
        Test that _compute_largest_free_space returns a reasonable value.
        For a 100x100 region with one small obstacle (20x20), the free area is about 9600.
        Depending on how the free space is split, the largest connected free space
        should be significantly larger than zero. Here, we check it is > 5000.
        """
        hd = HierarchicalDecomposition(self.region, self.obstacles)
        largest_free = hd._compute_largest_free_space(self.region, self.obstacles)
        self.assertGreater(largest_free, 5000, "Largest free space seems too small.")

    def test_has_single_connected_free_space(self):
        """
        Test _has_single_connected_free_space:
          - With the default obstacle, the largest free space is large => should return True.
          - With an obstacle that covers almost all of a small region, should return False.
        """
        hd = HierarchicalDecomposition(self.region, self.obstacles)
        self.assertTrue(hd._has_single_connected_free_space(self.region, self.obstacles),
                        "Expected connectivity with default obstacles.")
        # Create a small region nearly entirely covered by an obstacle.
        small_region = Polygon([(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)])
        almost_full_obs = Polygon([(0.1, 0.1), (9.9, 0.1), (9.9, 9.9), (0.1, 9.9), (0.1, 0.1)])
        hd2 = HierarchicalDecomposition(small_region, [almost_full_obs])
        self.assertFalse(hd2._has_single_connected_free_space(small_region, [almost_full_obs]),
                         "Expected connectivity check to fail for nearly full coverage.")

    def test_attempt_partition_degenerate_cut(self):
        """
        Test that _attempt_partition returns False if the division point is degenerate,
        i.e. it lies at or outside the region bounds.
        For a region [0,100] in x, a cut at x=0 is degenerate.
        """
        hd = HierarchicalDecomposition(self.region, self.obstacles)
        result = hd._attempt_partition(self.region, self.obstacles,
                                       axis='x', depth=0, division_point=0,
                                       subL=(self.region, self.obstacles),
                                       subR=(self.region, self.obstacles))
        self.assertFalse(result, "Degenerate partition should return False.")

    def test_mode_track_back(self):
        """
        Test that if _decompose produces no valid partition (returns False),
        the run() method in 'track_back' mode stores the entire region as a partition.
        We simulate this by subclassing HierarchicalDecomposition and forcing _decompose to return False.
        """

        class DummyHD(HierarchicalDecomposition):
            def _decompose(self, region, obstacles, depth):
                return False

        hd = DummyHD(self.region, self.obstacles, max_depth=3, mode="track_back")
        partitions = hd.run()
        self.assertEqual(len(partitions), 1, "Expected one partition from track_back fallback.")
        self.assertAlmostEqual(partitions[0][0].area, self.region.area,
                               "The stored partition should be the entire region.")


if __name__ == '__main__':
    unittest.main()
