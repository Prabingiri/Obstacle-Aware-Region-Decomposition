
import math
# from scipy.optimize import brentq, newton  # Commented out to remove dependency
import logging

# Configure logging minimally or comment it out if not needed
logging.basicConfig(level=logging.WARNING)

# def solve_for_root_brent(f, a, b, tol=1e-7, max_iter=100):
#     """
#     Finds a root of f(x)=0 in the interval [a,b] using Brent's method.
#     We keep only essential checks and minimal logging.
#     """
#     # logging.info(f"[Brent] Interval [{a}, {b}], tol={tol}")
#     fa, fb = f(a), f(b)
#     if fa * fb > 0:
#         raise ValueError("Brent: f(a)*f(b) must be <0 for a guaranteed sign change.")
#
#     c, fc = a, fa
#     d, e = b - a, b - a
#
#     for iteration in range(max_iter):
#         if fb * fc > 0:
#             c, fc = a, fa
#             d, e = b - a, b - a
#
#         if abs(fc) < abs(fb):
#             a, b, c = b, c, a
#             fa, fb, fc = fb, fc, fa
#
#         m = 0.5 * (c - b)
#         tol_act = 2 * tol * max(abs(b), 1.0)
#
#         # logging.debug(f"[Brent] Iter={iteration}, b={b}, f(b)={fb}, m={m}")
#         if abs(m) <= tol_act or fb == 0.0:
#             # logging.info(f"[Brent] Converged at {b}")
#             return b
#
#         if abs(e) >= tol_act and abs(fa) > abs(fb):
#             s = fb / fa
#             if a == c:
#                 # Secant
#                 p, q = 2 * m * s, 1 - s
#             else:
#                 q, r = fa / fc, fb / fc
#                 p = s * (2 * m * q * (q - r) - (b - a) * (r - 1))
#                 q = (q - 1) * (r - 1) * (s - 1)
#
#             if p > 0:
#                 q = -q
#             p = abs(p)
#             if 2 * p < min(3 * m * q - abs(tol_act * q), abs(e * q)):
#                 e, d = d, p / q
#             else:
#                 d, e = m, m
#         else:
#             d, e = m, m
#
#         a, fa = b, fb
#         b += d if abs(d) > tol_act else (-tol_act if m < 0 else tol_act)
#         fb = f(b)
#
#     # logging.warning(f"[Brent] Did not converge within {max_iter} iterations; returning {b}")
#     return b

def solve_for_root_brent(f, a, b, tol=1e-7, max_iter=100):
    """
    Finds a root of f(x)=0 in the interval [a,b] using Brent's method.

    This implementation ensures that b is the best approximation (smallest |f(b)|)
    and uses inverse quadratic interpolation when possible; otherwise it falls back
    to the bisection method.
    """
    fa = f(a)
    fb = f(b)
    if fa * fb > 0:
        raise ValueError("Brent: f(a)*f(b) must be <0 for a guaranteed sign change.")

    # Ensure that |f(b)| < |f(a)|
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = e = b - a

    for iteration in range(max_iter):
        # Check for convergence
        if abs(fb) < tol or abs(b - a) < tol:
            return b

        # Attempt inverse quadratic interpolation if possible
        if fa != fc and fb != fc:
            # Inverse quadratic interpolation:
            s = (a * fb * fc) / ((fa - fb) * (fa - fc)) \
                + (b * fa * fc) / ((fb - fa) * (fb - fc)) \
                + (c * fa * fb) / ((fc - fa) * (fc - fb))
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions to decide whether to accept s or fall back to bisection:
        if a < b:
            cond = (s < (3 * a + b) / 4 or s > b)
        else:
            cond = (s > (3 * a + b) / 4 or s < b)
        if cond or abs(s - b) >= abs(b - c) / 2:
            s = (a + b) / 2  # Bisection step
            d = e = b - a
        else:
            d = e

        fs = f(s)
        # Update the bracket: set c to the old b.
        c, fc = b, fb

        # Now choose the subinterval that contains the sign change:
        if fa * fs < 0:
            b = s
            fb = fs
        else:
            a = s
            fa = fs

        # Ensure that |f(b)| is the smaller of the two:
        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    # If the method did not converge within max_iter, return the current approximation.
    return b


def solve_for_root_newton_raphson(f, f_prime, x0, tol=1e-7, max_iter=100):
    """
    Newton-Raphson solver with minimal logging and standard fallback checks.
    """
    # logging.info(f"[Newton] x0={x0}, tol={tol}")
    x = x0
    for iteration in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        # logging.debug(f"[Newton] Iter={iteration}, x={x}, f(x)={fx}, f'(x)={fpx}")

        if abs(fx) < tol:
            # logging.info(f"[Newton] Converged at x={x}")
            return x
        if abs(fpx) < 1e-7:
            raise ValueError("Newton: derivative near zero => fails")

        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            # logging.info(f"[Newton] Converged at x={x_new}")
            return x_new
        x = x_new

    # logging.warning(f"[Newton] Not converged within {max_iter}, returning {x}")
    return x


# Commented out the SciPy-based methods to reduce redundancy:
# def solve_for_root_brent_python(f, a, b, tol=1e-7, max_iter=1000):
#     """
#     SciPy's brentq solver. Commented out to reduce redundancy.
#     """
#     return brentq(f, a, b, xtol=tol, maxiter=max_iter)
#
# def solve_for_root_newton_raphson_python(f, f_prime, x0, tol=1e-7, max_iter=1000):
#     """
#     SciPy's newton solver. Commented out to reduce redundancy.
#     """
#     return newton(f, x0=x0, fprime=f_prime, tol=tol, maxiter=max_iter)

def solve_for_root_with_defensive_newton_rhapson(
    f, f_prime, x0, bracket, tol=1e-7, max_iter_nr=100, max_iter_brent=100
):
    """
    Attempts Newton-Raphson first. If it fails, fallback to our custom Brent solver.
    """
    try:
        return solve_for_root_newton_raphson(f, f_prime, x0, tol=tol, max_iter=max_iter_nr)
    except Exception as e:
        # logging.warning(f"[Defensive] Newton-Raphson failed => fallback to Brent. {e}")
        return solve_for_root_brent(f, bracket[0], bracket[1], tol=tol, max_iter=max_iter_brent)
