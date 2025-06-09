"""
Ce fichier contient les fonctions qui permettent de calculer l'intégrale d'un élément simple de première espèce.
"""
import numpy as np


def calc_integral_type1(poly_up, poly_down, power, a, b):
    """
    Calcule l'intégrale d'une fonction de type:
        A
    –––––----
    (x - r)^n

    Avec A,r deux constantes et n une puissance positive entière.
    """
    if power == 1:   # Si la puissance vaut 1, on utilise la formule du log
        return poly_up[0] * (np.log(abs(b + poly_down[0])) - np.log(abs(a + poly_down[0])))
    else:  # Sinon, on utilise la formule plus générale de u' / u^n
        return poly_up[0] * (
            1/(1-power) * 1/((b + poly_down[0]) ** (power - 1)) - 1/(1-power) * 1/((a + poly_down[0]) ** (power - 1))
        )
