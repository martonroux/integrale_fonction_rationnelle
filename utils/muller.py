"""
Ce fichier contient les fonctions en lien avec la méthode de Muller.
Toutes les fonctions dans ce fichier ont été générées par Claude 3.7 Sonnet.
"""
import numpy as np
from numpy.polynomial import Polynomial, polynomial
import random

def mullers_method(poly, x0, x1, x2, max_iter=100, tol=1e-10):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 3.7 SONNET ------------
    """
    iterations = 0

    while iterations < max_iter:
        # Compute function values at the three points
        f0, f1, f2 = poly(x0), poly(x1), poly(x2)

        # Compute the differences
        h0 = x1 - x0
        h1 = x2 - x1

        # Avoid division by zero
        if abs(h0) < 1e-15 or abs(h1) < 1e-15:
            return x2, iterations

        # Compute the divided differences
        d0 = (f1 - f0) / h0
        d1 = (f2 - f1) / h1

        # Compute the coefficients of the quadratic approximation
        a = (d1 - d0) / (h1 + h0)
        b = a * h1 + d1
        c = f2

        # Calculate the next approximation
        if abs(a) < 1e-15:  # If the parabola is nearly a line
            if abs(b) < 1e-15:  # If it's a nearly constant function
                return x2, iterations
            x3 = x2 - c / b  # Linear case
        else:
            # Compute the discriminant
            discriminant = complex(b**2 - 4*a*c)
            sqrt_disc = np.sqrt(discriminant)

            # Choose the denominator that gives the smaller correction
            if abs(b + sqrt_disc) > abs(b - sqrt_disc):
                denominator = b + sqrt_disc
            else:
                denominator = b - sqrt_disc

            # Avoid division by zero
            if abs(denominator) < 1e-15:
                return x2, iterations

            # Compute the next approximation
            x3 = x2 - (2*c) / denominator

        # Check for convergence
        if abs(x3 - x2) < tol:
            return x3, iterations

        # Prepare for next iteration
        x0, x1, x2 = x1, x2, x3
        iterations += 1

    # If we reach here, we've exceeded max_iter
    return x2, iterations

def find_all_roots(coefficients, tol=1e-10, max_iter=100, attempts_per_root=5, verbose=False):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 3.7 SONNET ------------
    """
    # Create polynomial from coefficients
    poly = Polynomial(coefficients)

    if verbose:
        print(f"Finding roots of polynomial: {poly}")

    # Determine degree of the polynomial
    degree = poly.degree()

    roots = []

    for i in range(degree):
        root_found = False

        # Try different initial guesses
        for attempt in range(attempts_per_root):
            # Generate random complex initial guesses
            x0 = complex(random.uniform(-5, 5), random.uniform(-5, 5))
            x1 = complex(random.uniform(-5, 5), random.uniform(-5, 5))
            x2 = complex(random.uniform(-5, 5), random.uniform(-5, 5))

            # Apply Muller's method
            root, iterations = mullers_method(poly, x0, x1, x2, max_iter, tol)

            # Check if the root is valid
            if abs(poly(root)) < tol:
                # Clean up small real or imaginary parts
                if abs(root.real) < tol:
                    root = complex(0, root.imag)
                if abs(root.imag) < tol:
                    root = complex(root.real, 0)

                roots.append(root)

                if verbose:
                    print(f"Found root {i+1}: {format_complex(root)} (after {iterations} iterations)")

                divisor_coefs = [-root, 1]

                # Convert polynomial coefficients to complex if needed
                poly_coefs = poly.coef
                if isinstance(root, complex) and root.imag != 0:
                    poly_coefs = poly_coefs.astype(complex)

                # Perform polynomial division using numpy's polynomial division
                quotient_coefs, remainder = polynomial.polydiv(poly_coefs, divisor_coefs)
                poly = Polynomial(quotient_coefs)
                root_found = True
                break

        if not root_found:
            if verbose:
                print(f"Failed to find root {i+1} after {attempts_per_root} attempts")
            break

        # If the polynomial has been reduced to a constant or linear
        if poly.degree() <= 1:
            # If it's a linear polynomial, add its root
            if poly.degree() == 1:
                # For a linear polynomial ax + b, the root is -b/a
                coefs = poly.coef
                if abs(coefs[1]) > 1e-15:  # Ensure we don't divide by zero
                    linear_root = -coefs[0]/coefs[1]

                    # Clean up small real or imaginary parts
                    if abs(linear_root.real) < tol:
                        linear_root = complex(0, linear_root.imag)
                    if abs(linear_root.imag) < tol:
                        linear_root = complex(linear_root.real, 0)

                    roots.append(linear_root)

                    if verbose:
                        print(f"Found root {i+2} (linear): {format_complex(linear_root)}")
            break

    return roots

def format_complex(z, tol=1e-10):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 3.7 SONNET ------------
    """
    real_part = z.real if abs(z.real) >= tol else 0.0
    imag_part = z.imag if abs(z.imag) >= tol else 0.0

    if abs(imag_part) < tol:  # Real number
        return f"{real_part:.10g}"
    elif abs(real_part) < tol:  # Purely imaginary
        return f"{imag_part:.10g}j"
    elif imag_part > 0:  # Complex with positive imaginary part
        return f"{real_part:.10g} + {imag_part:.10g}j"
    else:  # Complex with negative imaginary part
        return f"{real_part:.10g} - {abs(imag_part):.10g}j"

def print_roots(roots):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 3.7 SONNET ------------
    """
    print("Roots found:")
    for i, root in enumerate(roots, 1):
        print(f"Root {i}: {format_complex(root)}")

def muller_find_roots(coefficients, tol=1e-10, max_iter=100, attempts=5, verbose=False):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 3.7 SONNET ------------

    Find all roots of a polynomial using Muller's method.

    Parameters:
    -----------
    coefficients : array-like
        Coefficients of the polynomial in ascending order (constant term first).
    tol : float, optional
        Tolerance for convergence.
    max_iter : int, optional
        Maximum number of iterations per root.
    attempts : int, optional
        Number of attempts with different initial guesses per root.
    verbose : bool, optional
        Whether to print additional information during execution.

    Returns:
    --------
    roots : list
        List of approximated roots.
    """
    roots = find_all_roots(coefficients, tol, max_iter, attempts, verbose)

    if verbose:
        print_roots(roots)

    return roots
