import math
import unittest
from shapely.geometry import Polygon, box, LineString, MultiPolygon, MultiLineString, GeometryCollection
from shapely.ops import unary_union
from unittest.mock import patch


from src.strip_perimeter import Strip, validate_and_fix_geometries


class TestStrip(unittest.TestCase):
    def setUp(self):
        """
        Create a simple region and a couple of obstacles.
        The region is a square from (0,0) to (100,100).
        Two obstacles are defined as boxes:
          - obstacle1: from (20,20) to (40,40)
          - obstacle2: from (60,60) to (80,80)
        For axis 'x', we expect the event coordinates to be:
           [0, 20, 40, 60, 80, 100]
        """
        self.region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        self.obstacle1 = box(20, 20, 40, 40)
        self.obstacle2 = box(60, 60, 80, 80)
        self.obstacles = [self.obstacle1, self.obstacle2]
        # Instantiate a Strip object using the x-axis.
        self.strip_x = Strip(self.region, self.obstacles, axis='x')

    def test_invalid_axis(self):
        """Test that providing an invalid axis raises a ValueError."""
        with self.assertRaises(ValueError):
            Strip(self.region, self.obstacles, axis='z')

    def test_validate_2d(self):
        """Test that _validate_2d drops Z-coordinates from a 3D geometry."""
        poly_3d = Polygon([
            (0, 0, 10),
            (100, 0, 10),
            (100, 100, 10),
            (0, 100, 10),
            (0, 0, 10)
        ])
        poly_2d = Strip._validate_2d(poly_3d)
        for coord in poly_2d.exterior.coords:
            self.assertEqual(len(coord), 2)

    def test_define_events(self):
        """
        For axis 'x', the events should include the region's min and max x
        and the x-coordinates of the obstacle exteriors.
        Expected events: [0, 20, 40, 60, 80, 100]
        """
        events = self.strip_x.events
        expected_events = [0.0, 20.0, 40.0, 60.0, 80.0, 100.0]
        self.assertEqual(events, expected_events)

    def test_create_strips(self):
        """
        Check that the number of strips equals len(events)-1 and that each
        strip (for axis 'x') is built as a box between consecutive x coordinates.
        """
        strips = self.strip_x.strips
        self.assertEqual(len(strips), len(self.strip_x.events) - 1)
        # Check the first strip
        c_prev, c_curr, strip_poly = strips[0]
        self.assertEqual(c_prev, 0.0)
        self.assertEqual(c_curr, 20.0)
        self.assertEqual(strip_poly.bounds, (0.0, 0.0, 20.0, 100.0))

    def test_compute_perimeters(self):
        """
        Test that the per-strip and cumulative perimeter dictionaries
        are populated and that cumulative perimeters are non-decreasing.
        """
        per_strip = self.strip_x.per_strip_perimeters
        cum_perim = self.strip_x.cumulative_perimeters
        expected_keys = [(0.0, 20.0), (20.0, 40.0), (40.0, 60.0), (60.0, 80.0), (80.0, 100.0)]
        self.assertEqual(sorted(per_strip.keys()), expected_keys)
        # Check that cumulative perimeters increase or remain equal
        cum_values = [cum_perim[k] for k in sorted(cum_perim.keys())]
        for i in range(1, len(cum_values)):
            self.assertGreaterEqual(cum_values[i], cum_values[i - 1])

    def test_compute_strip_perimeter(self):
        """
        Call compute_strip_perimeter on one of the strips and verify it returns a
        non-negative float.
        """
        _, _, strip_poly = self.strip_x.strips[0]
        perim = self.strip_x.compute_strip_perimeter(strip_poly)
        self.assertIsInstance(perim, float)
        self.assertGreaterEqual(perim, 0.0)

    def test_query_accumulated_perimeter(self):
        """
        For a coordinate that is an event (e.g., 20) the query should return
        the precomputed cumulative value. For a coordinate in between, the value
        should be at least as large as the previous event.
        """
        cp20 = self.strip_x.query_accumulated_perimeter(20.0)
        self.assertAlmostEqual(cp20, self.strip_x.cumulative_perimeters.get(20.0, cp20))
        cp30 = self.strip_x.query_accumulated_perimeter(30.0)
        self.assertGreaterEqual(cp30, cp20)

    def test_query_custom_strip_perimeter(self):
        """
        Test that query_custom_strip_perimeter returns a non-negative float.
        """
        custom_perim = self.strip_x.query_custom_strip_perimeter(20.0, 40.0, include_cumulative=True)
        self.assertIsInstance(custom_perim, float)
        self.assertGreaterEqual(custom_perim, 0.0)

    def test_calculate_total_obstacle_perimeter(self):
        """
        Ensure that the total obstacle perimeter is computed as a non-negative float.
        """
        total_perim = self.strip_x.calculate_total_obstacle_perimeter()
        self.assertIsInstance(total_perim, float)
        self.assertGreaterEqual(total_perim, 0.0)

    def test_calculate_region_diagonal(self):
        """
        For a region from (0,0) to (100,100), the diagonal should be approximately
        sqrt(100^2+100^2) ≈ 141.421356.
        """
        diag = self.strip_x.calculate_region_diagonal()
        expected = math.sqrt(100**2 + 100**2)
        self.assertAlmostEqual(diag, expected, places=5)

    def test_calculate_region_wcrt(self):
        """
        Verify that calculate_region_wcrt returns a float.
        """
        wcrt = self.strip_x.calculate_region_wcrt()
        self.assertIsInstance(wcrt, float)

    def test_calculate_region_diagonal_half(self):
        """
        For axis 'x', the half-diagonal should be sqrt(100^2 + (100/2)^2)
        ≈ sqrt(10000+2500)= sqrt(12500) ≈ 111.8034.
        """
        half_diag = self.strip_x.calculate_region_diagonal_half()
        expected = math.sqrt(100**2 + (100 / 2)**2)
        self.assertAlmostEqual(half_diag, expected, places=4)

    def test_calculate_target_wcrt(self):
        """Verify that calculate_target_wcrt returns a float."""
        target = self.strip_x.calculate_target_wcrt()
        self.assertIsInstance(target, float)

    def test_calculate_target_wcrt_dynamic(self):
        """Verify that calculate_target_wcrt_dynamic returns a float."""
        target_dynamic = self.strip_x.calculate_target_wcrt_dynamic()
        self.assertIsInstance(target_dynamic, float)

    def test_calculate_diagonal_at_coordinate(self):
        """
        For axis 'x' and coordinate 50, the diagonal should be:
            sqrt( (region height)^2 + (50 - region_min_x)^2 )
        For our region, that is sqrt(100^2 + 50^2) ≈ 111.8034.
        """
        diag_at_50 = self.strip_x.calculate_diagonal_at_coordinate(50.0)
        expected = math.sqrt(100**2 + 50**2)
        self.assertAlmostEqual(diag_at_50, expected, places=4)

    def test_calculate_wcrt_at_strip(self):
        """
        Test that calculate_wcrt_at_strip (using one of the strip polygons)
        returns a float.
        """
        _, _, strip_poly = self.strip_x.strips[0]
        wcrt_strip = self.strip_x.calculate_wcrt_at_strip(strip_poly)
        self.assertIsInstance(wcrt_strip, float)

    def test_query_wcrt_at_coordinate(self):
        """
        Verify that query_wcrt_at_coordinate returns a float.
        """
        wcrt_coord = self.strip_x.query_wcrt_at_coordinate(50.0)
        self.assertIsInstance(wcrt_coord, float)

    def test_calculate_wcrt_left_right(self):
        """
        Check that both calculate_wcrt_left and calculate_wcrt_right return floats.
        """
        left = self.strip_x.calculate_wcrt_left(40.0)
        right = self.strip_x.calculate_wcrt_right(40.0)
        self.assertIsInstance(left, float)
        self.assertIsInstance(right, float)

    def test_calculate_diagonal_right(self):
        """Verify that calculate_diagonal_right returns a float."""
        diag_right = self.strip_x.calculate_diagonal_right(40.0)
        self.assertIsInstance(diag_right, float)

    def test_calculate_total_obstacle_area(self):
        """
        For obstacles defined as:
          - obstacle1: box(20,20,40,40) → area 400
          - obstacle2: box(60,60,80,80) → area 400
        The total area should be 800.
        """
        total_area = self.strip_x.calculate_total_obstacle_area()
        expected_area = self.obstacle1.area + self.obstacle2.area
        self.assertAlmostEqual(total_area, expected_area, places=5)

    def test_is_edge_collinear_with_coord(self):
        """
        Test that is_edge_collinear_with_coord correctly identifies collinearity.
        """
        # Create a vertical line at x=10.
        line = LineString([(10, 5), (10, 15)])
        self.assertTrue(Strip.is_edge_collinear_with_coord(line, 10, 'x'))
        # A line not perfectly vertical:
        line2 = LineString([(10, 5), (11, 15)])
        self.assertFalse(Strip.is_edge_collinear_with_coord(line2, 10, 'x'))



if __name__ == '__main__':
    unittest.main()
