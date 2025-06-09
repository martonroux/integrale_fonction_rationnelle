"""
Ce fichier contient des fonctions utiles sans catégorie particulière.
"""
import numpy as np


def unique_with_epsilon(numbers, epsilon=1e-6):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 4.0 SONNET ------------

    Permet de trouver les racines multiples à epsilon près (les valeurs ne sont pas exactes), et retourne leur multiplicité.
    """
    if not numbers:
        return [], []

    numbers = np.array(numbers)
    unique_values = []
    counts = []

    for num in numbers:
        if abs(num.imag) < epsilon:  # If imaginary part too small, discard it
            num = num.real
        # Check if this number is close to any existing unique value
        found_match = False
        for i, unique_val in enumerate(unique_values):
            if abs(num - unique_val) <= epsilon:
                counts[i] += 1
                found_match = True
                break

        # If no match found, add as new unique value
        if not found_match and num.imag >= 0:  # Skip conjugates
            unique_values.append(np.round(num, 7))
            counts.append(1)

    return unique_values, counts


def solve_linear_system(A, b, method='solve'):
    """
    ------------ GÉNÉRÉ PAR CLAUDE 4.0 SONNET ------------

    Permet de résoudre le système linéaire d'identification des coefficients.

    Solves the linear system Ax = b.

    Parameters:
    -----------
    A : array-like, shape (m, n)
        Coefficient matrix
    b : array-like, shape (m,)
        Right-hand side vector
    method : str, default 'solve'
        Method to use: 'solve', 'lstsq', or 'pinv'

    Returns:
    --------
    dict: Contains solution and additional info
        'solution': the solution vector x
        'residual': residual norm (for overdetermined systems)
        'rank': rank of matrix A
        'singular': True if matrix is singular
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    result = {
        'solution': None,
        'residual': None,
        'rank': np.linalg.matrix_rank(A),
        'singular': False
    }

    try:
        if method == 'solve':
            # For square, non-singular systems
            if A.shape[0] != A.shape[1]:
                method = 'lstsq'
            else:
                result['solution'] = np.linalg.solve(A, b)

        if method == 'lstsq':
            # For overdetermined or underdetermined systems
            solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            result['solution'] = solution
            result['residual'] = residuals[0] if len(residuals) > 0 else 0
            result['rank'] = rank

        elif method == 'pinv':
            # Using pseudoinverse (Moore-Penrose)
            result['solution'] = np.linalg.pinv(A) @ b
            result['residual'] = np.linalg.norm(A @ result['solution'] - b)

    except np.linalg.LinAlgError:
        result['singular'] = True
        solution, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        result['solution'] = solution
        result['residual'] = residuals[0] if len(residuals) > 0 else 0
        result['rank'] = rank

    return result
