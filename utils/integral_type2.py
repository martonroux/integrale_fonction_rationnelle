"""
Ce fichier contient toutes les fonctions qui servent à calculer l'intégrale des éléments simples de seconde espèce.
"""
import numpy as np


def calc_integral_cosine_pow_n(power: int, interval: list):
    """
    Calcule l'intégrale de cosinus à une puissance n.
    Cette fonction est récursive: elle s'appelle elle-même, pour simuler le comportement d'une suite.
    On nomme I(n) cette suite
    """
    if power == 0:  # I(0)
        return interval[1] - interval[0]
    if power == 1:  # I(1)
        return np.sin(interval[1]) - np.sin(interval[0])

    left_part = (
                (np.sin(interval[1]) * (np.cos(interval[1]) ** (power - 1)))
                - (np.sin(interval[0]) * (np.cos(interval[0]) ** (power - 1)))
        ) / power

    right_part = ((power - 1) / power) * calc_integral_cosine_pow_n(power - 2, interval)  # I(n-2)

    return left_part + right_part


def calc_type2_power_simple(power, poly, a, b):
    """
    Calcule l'intégrale d'une fraction de la sorte:
           1
    --------------
    (x^2 + cx + d)^n

    Où c, d sont deux constantes réelles, et n un entier positif >= 2.
    """
    alpha = poly[1] / 2  # Raccourcis pour simplifier le calcul visuellement
    beta = poly[2] - alpha ** 2

    upper_bound = np.arctan((b + alpha) / (beta ** 0.5))  # Borne supérieure de l'intégrale
    lower_bound = np.arctan((a + alpha) / (beta ** 0.5))  # Borne inférieure de l'intégrale

    return beta ** 0.5 / (beta ** power) * calc_integral_cosine_pow_n(2 * power - 2, [lower_bound, upper_bound])


def calc_type2_power(power, poly_up, poly_down, a, b):
    """
    Calcule l'intégrale d'une fraction de la sorte:
        Ax + B
    --------------
    (x^2 + cx + d)^n

    Où A, B, c, d sont des constantes réelles, et n un entier positif >= 2.
    """
    left_integral = poly_up[1] / 2 * (
        (
                1/(1-power) * 1/(b ** 2 + poly_down[1] * b + poly_down[0]))
                - 1/(1-power) * 1/(a ** 2 + poly_down[1] * a + poly_down[0])
    )
    right_integral = (poly_up[0] - poly_up[1] * poly_down[1] / 2) * calc_type2_power_simple(power, poly_down, a, b)
    return left_integral + right_integral


def calc_type2_no_power(poly_up, poly_down, a, b):
    """
    Calcule l'intégrale d'une fraction de la sorte:
        Ax + B
    --------------
    (x^2 + cx + d)

    Où A, B, c, d sont des constantes réelles.
    """
    alpha = poly_down[1] / 2
    beta = poly_down[0] - (poly_down[1] / 2) ** 2
    left_integral = poly_up[1] / 2 * (np.log(abs(b**2 + poly_down[1] * b + poly_down[0])) - np.log(abs(a**2 + poly_down[1] * a + poly_down[0])))

    right_integral = (poly_up[0] - poly_down[1] * poly_up[1] / 2) * 1/np.sqrt(beta) * (
            np.arctan((b + alpha) / (np.sqrt(beta)))
            - np.arctan((a + alpha) / (np.sqrt(beta)))
    )
    return left_integral + right_integral


def calc_integral_type2(poly_up, poly_down, power, a, b):
    """
    Calcule l'intégrale d'une fraction de la sorte:
        Ax + B
    --------------
    (x^2 + cx + d)^n

    Où A, B, c, d sont des constantes réelles, et n un entier positif.
    Cette fonction est une fonction "pilote" qui appelle celles déclarées plus haut.
    """
    if power == 1:
        return calc_type2_no_power(poly_up, poly_down, a, b)
    return calc_type2_power(power, poly_up, poly_down, a, b)
