import math
import unittest

# Import the numerical solvers.
# Adjust the import paths as needed.
from src.numerical_solution import (
    solve_for_root_brent,
    solve_for_root_newton_raphson,
    solve_for_root_with_defensive_newton_rhapson
)

class TestNumericalSolution(unittest.TestCase):

    def test_brent_convergence(self):
        """
        Test Brent's method on f(x) = x^2 - 2, whose root is sqrt(2) ≈ 1.41421356.
        **Expected Behavior:**
        The correct implementation should return a value close to sqrt(2).
        (Currently, the implementation returns 1 due to an early termination caused
         by an incorrect swap of variables. Once fixed, this test should pass.)
        """
        def f(x):
            return x * x - 2

        root = solve_for_root_brent(f, 1, 2)
        self.assertAlmostEqual(root, math.sqrt(2), places=6)

    def test_brent_no_sign_change(self):
        """
        Test that Brent's method raises a ValueError when the function does not change sign.
        For f(x) = x^2 + 1, f(0)=1 and f(1)=2 (no sign change).
        """
        def f(x):
            return x * x + 1

        with self.assertRaises(ValueError):
            solve_for_root_brent(f, 0, 1)

    def test_newton_convergence(self):
        """
        Test Newton-Raphson on f(x) = x^2 - 2 with derivative f'(x) = 2*x.
        Starting from x0 = 1.0, the method should converge to sqrt(2).
        """
        def f(x):
            return x * x - 2

        def f_prime(x):
            return 2 * x

        root = solve_for_root_newton_raphson(f, f_prime, 1.0)
        self.assertAlmostEqual(root, math.sqrt(2), places=6)

    def test_newton_derivative_near_zero(self):
        """
        Test that Newton-Raphson raises a ValueError when the derivative is near zero.
        We use a constant function f(x)=1 (so that f'(x)=0 everywhere) and an initial guess.
        """
        def f(x):
            return 1  # Constant function; f(x) never reaches zero.

        def f_prime(x):
            return 0  # Derivative is zero.

        with self.assertRaises(ValueError):
            solve_for_root_newton_raphson(f, f_prime, 0.5)

    def test_defensive_newton_newton_success(self):
        """
        Test the defensive solver on a function where Newton-Raphson succeeds.
        Using f(x) = x^2 - 2, starting at x0 = 1.0 with bracket [1,2] should return sqrt(2).
        """
        def f(x):
            return x * x - 2

        def f_prime(x):
            return 2 * x

        root = solve_for_root_with_defensive_newton_rhapson(f, f_prime, 1.0, bracket=[1, 2])
        self.assertAlmostEqual(root, math.sqrt(2), places=6)

    def test_defensive_newton_newton_failure(self):
        """
        Test the defensive solver on a function where Newton-Raphson fails.
        For f(x) = x^3 (with derivative f'(x) = 3*x^2), starting at x0 = 0.0 results in f'(0) = 0.
        The Newton method should fail and the solver should fall back to Brent's method.
        The root of x^3 is 0.
        """
        def f(x):
            return x ** 3

        def f_prime(x):
            return 3 * x * x

        # The bracket [-1, 1] guarantees a sign change since f(-1) = -1 and f(1) = 1.
        root = solve_for_root_with_defensive_newton_rhapson(f, f_prime, 0.0, bracket=[-1, 1])
        self.assertAlmostEqual(root, 0.0, places=6)

if __name__ == '__main__':
    unittest.main()
