import math
import unittest
from shapely.geometry import Polygon, box
from src.optimal_axis_selection import OptimalAxisSelection

class TestOptimalAxisSelection(unittest.TestCase):
    def setUp(self):
        # Create a simple square region [0,100]x[0,100]
        self.region = Polygon([(0, 0), (100, 0), (100, 100), (0, 100), (0, 0)])
        # Create one dummy obstacle (a 20x20 square).
        self.obstacles = [box(20, 20, 40, 40)]
        # Create an OptimalAxisSelection instance with default numerical_method.
        self.optimal = OptimalAxisSelection(self.region, self.obstacles, numerical_method='newton')

    def test_default_user_metric(self):
        # If user_metric is not provided or is invalid, it should default to "NWCRT".
        opt1 = OptimalAxisSelection(self.region, self.obstacles, user_metric="WRONG")
        self.assertEqual(opt1.user_metric, "NWCRT")
        opt2 = OptimalAxisSelection(self.region, self.obstacles)
        self.assertEqual(opt2.user_metric, "NWCRT")

    def test_square_measure(self):
        # For a square, width == height, so _square_measure should return 1.0.
        ratio = self.optimal._square_measure(self.region)
        self.assertAlmostEqual(ratio, 1.0)
        # For a rectangle 100x50, the ratio should be 100/50 = 2.0.
        rect = Polygon([(0, 0), (100, 0), (100, 50), (0, 50), (0, 0)])
        ratio_rect = self.optimal._square_measure(rect)
        self.assertAlmostEqual(ratio_rect, 2.0)

    def test_calculate_squareness_metrics(self):
        # When both left and right are squares (sq=1), then
        # msdu = 1 / (0.5*(0^2+0^2)+eps) = 1/eps (with eps=1e-9).
        _, msdu = self.optimal.calculate_squareness_metrics(1.0, 1.0)
        self.assertAlmostEqual(msdu, 1.0/1e-9, places=3)
        # Test with different values:
        # For example, if sq_left = 2 and sq_right = 0.5:
        expected = 1/ (0.5*((2-1)**2 + (0.5-1)**2) + 1e-9)
        _, msdu2 = self.optimal.calculate_squareness_metrics(2.0, 0.5)
        self.assertAlmostEqual(msdu2, expected, places=3)

    def test_evaluate_axis_returns_metrics(self):
        """
        Test that evaluate_axis returns a dictionary with keys:
        "NWCRT", "_MSDU", "_division_point", "_subregion_left", "_subregion_right".
        (The actual values come from the underlying Strip and ObstacleAwareDivider,
        which are assumed to work as tested previously.)
        """
        metrics = self.optimal.evaluate_axis('x')
        for key in ["NWCRT", "_MSDU", "_division_point", "_subregion_left", "_subregion_right"]:
            self.assertIn(key, metrics)
        self.assertIsInstance(metrics["NWCRT"], float)
        self.assertIsInstance(metrics["_MSDU"], float)

    def test_select_best_axis_no_tie(self):
        """
        Simulate a scenario where the 'x' axis has a smaller NWCRT than the 'y' axis.
        In that case, select_best_axis should choose 'x'.
        """
        # Monkey-patch evaluate_axis to return controlled metrics.
        def eval_axis_x(axis):
            return {
                "NWCRT": 0.2,
                "_MSDU": 10,
                "_division_point": 50,
                "_subregion_left": ("left_x", "obs_left_x"),
                "_subregion_right": ("right_x", "obs_right_x")
            }
        def eval_axis_y(axis):
            return {
                "NWCRT": 0.3,
                "_MSDU": 20,
                "_division_point": 55,
                "_subregion_left": ("left_y", "obs_left_y"),
                "_subregion_right": ("right_y", "obs_right_y")
            }
        self.optimal.evaluate_axis = lambda axis: eval_axis_x(axis) if axis == 'x' else eval_axis_y(axis)
        best_axis, overall_metrics, best_div_point, best_sub_left, best_sub_right = self.optimal.select_best_axis()
        self.assertEqual(best_axis, 'x')
        self.assertEqual(best_div_point, 50)
        self.assertEqual(best_sub_left, ("left_x", "obs_left_x"))
        self.assertEqual(best_sub_right, ("right_x", "obs_right_x"))

    def test_select_best_axis_tie_breaker(self):
        """
        Simulate a tie in NWCRT between axes 'x' and 'y'. In this case, the axis with
        the larger _MSDU should be chosen.
        """
        def eval_axis_x(axis):
            return {
                "NWCRT": 0.25,
                "_MSDU": 15,
                "_division_point": 50,
                "_subregion_left": ("left_x", "obs_left_x"),
                "_subregion_right": ("right_x", "obs_right_x")
            }
        def eval_axis_y(axis):
            return {
                "NWCRT": 0.25,
                "_MSDU": 20,
                "_division_point": 60,
                "_subregion_left": ("left_y", "obs_left_y"),
                "_subregion_right": ("right_y", "obs_right_y")
            }
        self.optimal.evaluate_axis = lambda axis: eval_axis_x(axis) if axis == 'x' else eval_axis_y(axis)
        best_axis, overall_metrics, best_div_point, best_sub_left, best_sub_right = self.optimal.select_best_axis()
        # Tie-breaker: larger _MSDU wins, so axis 'y' should be chosen.
        self.assertEqual(best_axis, 'y')
        self.assertEqual(best_div_point, 60)
        self.assertEqual(best_sub_left, ("left_y", "obs_left_y"))
        self.assertEqual(best_sub_right, ("right_y", "obs_right_y"))

if __name__ == '__main__':
    unittest.main()
