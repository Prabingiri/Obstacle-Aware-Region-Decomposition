import math
import unittest
from shapely.geometry import Polygon, box, LineString, GeometryCollection
from shapely.ops import unary_union
from shapely import make_valid
from unittest.mock import patch

# Adjust the import paths as needed.
# Here we assume the module is located at src/obstacle_aware_divider.py.
from src.obstacle_aware_divider import ObstacleAwareDivider, validate_and_fix_geometries


# -------------------------------
# Dummy Strip Processor
# -------------------------------
class DummyStripProcessor:
    """
    A dummy stand-in for a real Strip processor.
    Provides fixed values for a convex square region, one obstacle,
    and two strips along the x-axis.
    """

    def __init__(self, axis='x'):
        self.axis = axis
        # Define the region as a 100x100 square.
        self.region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        # Define one obstacle: a 20x20 square inside the left half.
        self.obstacles = [box(20, 20, 40, 40)]
        # Define two strips along the x-axis:
        # First strip from x=0 to x=50, second strip from x=50 to x=100.
        self.strips = [
            (0.0, 50.0, Polygon([(0, 0), (50, 0), (50, 100), (0, 100), (0, 0)])),
            (50.0, 100.0, Polygon([(50, 0), (100, 0), (100, 100), (50, 100), (50, 0)]))
        ]

    def calculate_region_diagonal(self):
        # Diagonal of 100x100 square.
        return math.sqrt(100 ** 2 + 100 ** 2)

    def calculate_total_obstacle_perimeter(self):
        # A 20x20 square has a perimeter of 80.
        return 80.0

    def calculate_target_wcrt(self):
        # Return a fixed target value.
        return 150.0

    def calculate_diagonal_at_coordinate(self, coord):
        # For axis 'x': diagonal from (0,0) to (coord,100).
        return math.sqrt(100 ** 2 + (coord) ** 2)

    def query_accumulated_perimeter(self, coord):
        # For simplicity, return 40 for coord ≤ 50, and 80 otherwise.
        return 40.0 if coord <= 50 else 80.0

    def calculate_diagonal_right(self, coord):
        # For axis 'x': diagonal from (coord,0) to (100,100).
        return math.sqrt(100 ** 2 + (100 - coord) ** 2)

    def compute_strip_perimeter(self, strip_geometry):
        # Return a fixed value for testing.
        return 10.0

    def calculate_wcrt_left(self, coord):
        # WCRT_left = diagonal_at_coordinate(coord) + 0.5 * accumulated perimeter.
        return self.calculate_diagonal_at_coordinate(coord) + 0.5 * self.query_accumulated_perimeter(coord)

    def calculate_wcrt_right(self, coord):
        # WCRT_right = diagonal_right(coord) + 0.5 * (total obstacle perimeter − accumulated perimeter).
        return self.calculate_diagonal_right(coord) + 0.5 * (
                    self.calculate_total_obstacle_perimeter() - self.query_accumulated_perimeter(coord))


# -------------------------------
# Unit Tests for ObstacleAwareDivider
# -------------------------------
class TestObstacleAwareDivider(unittest.TestCase):
    def setUp(self):
        # Create a dummy strip processor (with axis 'x') and initialize the divider.
        self.dsp = DummyStripProcessor(axis='x')
        self.divider = ObstacleAwareDivider(self.dsp, method='brent')

    def test_g_function(self):
        """
        Test that g(cut_coord) returns WCRT_left − WCRT_right.
        """
        coord = 75.0
        expected = self.dsp.calculate_wcrt_left(coord) - self.dsp.calculate_wcrt_right(coord)
        self.assertAlmostEqual(self.divider.g(coord), expected)

    def test_g_prime_function(self):
        """
        Test that g_prime (the numerical derivative) returns a float.
        """
        coord = 75.0
        gp = self.divider.g_prime(coord)
        self.assertIsInstance(gp, float)

    def test_find_strip_of_interest(self):
        """
        With our dummy strips, the second strip (from 50 to 100) is where
        WCRT_left > WCRT_right; the method should return that strip.
        """
        strip_interest = self.divider.find_strip_of_interest()
        self.assertIsNotNone(strip_interest)
        coord_prev, coord_curr, _ = strip_interest
        self.assertEqual(coord_prev, 50.0)
        self.assertEqual(coord_curr, 100.0)

    def test_determine_case_for_strip_no_obstacle(self):
        """
        If there are no obstacles, determine_case_for_strip should return 1.
        """
        dsp_no_obs = DummyStripProcessor(axis='x')
        dsp_no_obs.obstacles = []  # Remove obstacles.
        divider_no_obs = ObstacleAwareDivider(dsp_no_obs, method='brent')
        # For any strip (here, first strip), expect case 1.
        strip = dsp_no_obs.strips[0]
        self.assertEqual(divider_no_obs.determine_case_for_strip(strip), 1)

    def test_determine_case_for_strip_degeneracy(self):
        """
        Simulate degeneracy by using an obstacle whose vertical edge is exactly at the strip boundary.
        For axis 'x', an obstacle with an edge at x=50 (the upper boundary of the first strip)
        should yield case 3.

        (Ensure that your _extract_edges implementation returns segments for LineString
         boundaries so that such degeneracy is detected.)
        """
        degenerate_obs = Polygon([(50, 10), (50, 20), (60, 20), (60, 10), (50, 10)])
        dsp_deg = DummyStripProcessor(axis='x')
        dsp_deg.obstacles = [degenerate_obs]
        divider_deg = ObstacleAwareDivider(dsp_deg, method='brent')
        # Use the first strip, whose upper boundary is x=50.
        strip = dsp_deg.strips[0]
        self.assertEqual(divider_deg.determine_case_for_strip(strip), 3)

    @patch('src.obstacle_aware_divider.solve_for_root_brent', return_value=55.0)
    def test_handle_case_1(self, mock_brent):
        """
        Test handle_case_1 returns the numerical method’s result if a sign change is detected.
        """
        result = self.divider.handle_case_1(40.0, 60.0, -1.0, 1.0)
        self.assertEqual(result, 55.0)

    @patch('src.obstacle_aware_divider.solve_for_root_brent', return_value=65.0)
    def test_handle_case_2(self, mock_brent):
        """
        Test handle_case_2 behaves similarly.
        """
        result = self.divider.handle_case_2(40.0, 60.0, -1.0, 1.0)
        self.assertEqual(result, 65.0)

    @patch.object(ObstacleAwareDivider, 'apply_numerical_method', return_value=70.0)
    def test_handle_case_3(self, mock_apply):
        """
        Test handle_case_3.
        We temporarily override g so that g(c_j - delta) <= 0; in that case, the method should return c_j.
        """
        original_g = self.divider.g
        self.divider.g = lambda x: -0.1 if abs(x - 60.0) < 1e-3 else 1.0
        result = self.divider.handle_case_3(40.0, 60.0, -1.0, 1.0)
        self.assertEqual(result, 60.0)
        self.divider.g = original_g

    @patch('src.obstacle_aware_divider.solve_for_root_brent', return_value=80.0)
    def test_apply_numerical_method(self, mock_brent):
        """
        Test that apply_numerical_method returns the value from the numerical solver.
        """
        result = self.divider.apply_numerical_method(40.0, 60.0)
        self.assertEqual(result, 80.0)

    def test_divide_region(self):
        """
        Test that divide_region splits the region and obstacles correctly.
        For the dummy setup (a 100x100 region with one obstacle from (20,20) to (40,40)),
        dividing at x = 50 should yield:
          - The left subregion contains the obstacle.
          - The right subregion contains no obstacles.
        """
        (R_left, left_obs), (R_right, right_obs) = self.divider.divide_region(50.0)
        self.assertFalse(R_left.is_empty)
        self.assertFalse(R_right.is_empty)
        self.assertEqual(len(left_obs), 1)
        self.assertEqual(len(right_obs), 0)

    def test_extract_polygonal_part(self):
        """
        Test that _extract_polygonal_part returns a Polygon when given a Polygon,
        extracts polygonal parts from a GeometryCollection, and returns None for non-polygon types.
        """
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
        result = self.divider._extract_polygonal_part(poly)
        self.assertEqual(result, poly)

        ls = LineString([(0, 0), (1, 1)])
        gc = GeometryCollection([poly, ls])
        result_gc = self.divider._extract_polygonal_part(gc)
        self.assertTrue(result_gc.equals(poly))

        ls_only = LineString([(0, 0), (1, 1)])
        result_ls = self.divider._extract_polygonal_part(ls_only)
        self.assertIsNone(result_ls)


if __name__ == '__main__':
    unittest.main()
