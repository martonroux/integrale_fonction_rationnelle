"""
Ce fichier est le fichier principal. Contient la logique du programme.
"""
from numpy.polynomial import Polynomial
import numpy as np
from utils.muller import muller_find_roots
from utils.integral_type2 import calc_integral_type2
from utils.integral_type1 import calc_integral_type1
from utils.utils import solve_linear_system
from utils.utils import unique_with_epsilon


def get_other_roots(roots, counts):
    """
    Pour identifier les coefficients, on rammène tous les éléments simples sous le même dénominateur.
    Cette fonction retourne le polynôme par lequel multiplier le haut et le bas de l'élément simple que l'on étudie
    actuellement.

    Par exemple, si on a:
      A       B        C
    ----- + ----- + -------
    x - 2   x - 4   (x-1)^2

    Et que on travaillait actuellement sur le 3ème élément simple, cette fonction retournerait:
    (x-2) * (x-4) = x^2 - 6x + 8

    Soit le produit des autres dénominateurs.
    On l'utiliserait de cette façon:

       C * (x^2 - 6x + 8)
    ------------------------
    (x-1)^2 * (x^2 - 6x + 8)

    """
    if len(roots) == 0 or roots[0].imag < 0:  # On passe le conjugué des racines complexes pour éviter les doublons.
        return Polynomial([1])

    if roots[0].imag == 0:
        return (Polynomial([-roots[0].real, 1]) ** counts[0]) * get_other_roots(roots[1:], counts[1:])
    return (Polynomial([roots[0].real ** 2 + roots[0].imag ** 2, -roots[0].real * 2, 1]) ** counts[0]) * get_other_roots(roots[1:], counts[1:])


def get_polys_simple_element(index: int, roots, counts, max_degree):
    """
    Une des fonctions principales. Retourne une matrice selon ce formatage :

    - Chaque colonne correspond à une des constantes au numérateur des éléments simples.
    - Chaque ligne correspond à un des x^i.

    Par exemple, avec :
      A       B         Ax - 4A         Bx - 2B
    ----- + ----- = -------------- + --------------
    x - 2   x - 4   (x - 2)(x - 4)   (x - 2)(x - 4)

    On aurait :
       A   B
    [ -4, -2 ] ← x^0
    [  1,  1 ] ← x^1

    Ceci serait le résultat retourné par cette fonction.
    """
    root = roots[index]  # On sélectionne la racine actuelle.
    count = counts[index]  # La multiplicité de la racine. Pour une racine double, on aura deux éléments simples :
    # Le premier avec une puissance de 1 au dénominateur
    # Le second avec une puissance de 2 au dénominateur.
    # La (ou les, dans le cas d'une racine complexe) constante(s) qui se trouvent au numérateur seront uniques
    # pour chaque puissance au dénominateur.
    # Exemple :
    #   P(x)
    # --------
    # (x - 2)^2
    # Ici, la décomposition en éléments simples se fait comme ceci :
    #
    #   B        C
    # ----- + -------
    # x - 2   (x-2)^2
    #
    # Donc les constantes B et C sont uniques. Comme la multiplicité de la racine est de 2, on a 2 constantes différentes.
    # D'où l'importance de cette variable.

    if root.imag < 0:  # On passe le conjugué des racines complexes pour éviter les doublons.
        return None

    if root.imag != 0:  # Si la racine et complexe, élément simple de seconde espère
        base_poly = [root.imag**2 + root.real**2, -2 * root.real, 1]  # Le polynôme en dénominateur de l'élément simple
        num_vars = count * 2  # Les éléments simples de seconde espèce ont 2 constantes au numérateur (Ax + B).
    else:  # Sinon, élément simple de première espère
        base_poly = [-root.real, 1]  # Le polynôme en dénominateur de l'élément simple
        num_vars = count

    poly = Polynomial(base_poly)
    matrix = np.zeros([max_degree + 1, num_vars])  # La matrice que l'on remplit de 0.
    # Le degré maximum de x^i correspond au degré du dénominateur de la fraction rationnelle d'origine.
    # Donc si le dénominateur est de degré 3, il y aura 3 lignes.

    for i in range(0, count):
        # On élève progressivement le polynôme à une puissance, tout en la multipliant par les polynômes
        # des autres éléments simples pour ramener au même dénominateur.
        elevated_poly = poly ** (count - (i + 1)) * get_other_roots(roots[:index] + roots[index + 1:], counts[:index] + counts[index + 1:])

        # On regarde les coefficients devant chaque x^i du nouveau polynôme.
        for j, coeff in enumerate(elevated_poly):
            matrix[j, i] = coeff

        # Si c'est un élement simple de première espèce, on s'arrête là.
        if root.imag == 0:
            continue

        # Sinon, on refait l'opération pour A (si on avait Ax + B au numérateur).
        for j, coeff in enumerate(elevated_poly):
            matrix[j + 1, i + count] = coeff
    return matrix


def get_floor_polynomial(poly_up: list, poly_down: list):
    """
    Cette fonction calcule la division euclidienne entre deux polynômes.
    Permet de calculer la partie entière à intégrer séparément.
    """
    if len(poly_up) < len(poly_down):
        return [], poly_up
    quotient, remainder = np.polydiv(poly_up[::-1], poly_down[::-1])
    return list(quotient[::-1]), list(remainder[::-1])


def integrate_floored_polynomial(poly: list, a, b):
    """
    Cette fonction calcule l'intégrale d'un polynôme entre deux bornes.
    Utilisé pour calculer l'intégrale de la partie entière.
    """
    integral = np.polyint(poly[::-1])[::-1]
    left_part = sum([integral[i] * (b ** i) for i in range(len(integral))])
    right_part = sum([integral[i] * (a ** i) for i in range(len(integral))])

    return left_part - right_part


def calc_integral(poly_up: list, poly_down: list, a, b):
    """
    Ceci est la fonction principale. Elle calcule l'intégrale d'une fonction rationnelle entre deux points a et b.
    """
    # Extraire la partie entière
    floored_poly_up, rest_poly_up = get_floor_polynomial(poly_up, poly_down)

    roots = muller_find_roots(poly_down, verbose=False)  # On récupère les racines du dénominateur
    unique, count = unique_with_epsilon(roots)  # On récupère les racines uniques avec leur multiplicité
    whole_matrix = None

    for i in range(len(unique)):  # Pour chaque racine unique, on calcule la matrice des coefficients
        matrix = get_polys_simple_element(i, unique, count, len(poly_down) - 1)
        if matrix is None:
            continue
        if whole_matrix is None:
            whole_matrix = matrix
        else:
            whole_matrix = np.concatenate((whole_matrix, matrix), axis=1)  # On ajoute progressivement à la matrice existante

    # On calcule la solution au système d'identification des coefficients pour trouver les constantes au numérateur
    # des éléments simples.
    constants = solve_linear_system(whole_matrix, rest_poly_up + [0 for _ in range(len(poly_down) - len(rest_poly_up))])['solution']
    integral = 0
    constant_idx = 0
    for i in range(len(unique)):  # Pour chaque élément simple, on calcule son intégrale, et on l'ajoute au résultat
        if unique[i].imag == 0:
            for j in range(1, count[i] + 1):
                integral += calc_integral_type1([constants[constant_idx]], [-unique[i].real, 1], j, a, b)
                constant_idx += 1
        else:
            for j in range(1, count[i] + 1):
                integral += calc_integral_type2(
                    [constants[constant_idx], constants[constant_idx + 1]],
                    [unique[i].real ** 2 + unique[i].imag ** 2, -unique[i].real * 2, 1],
                    j,
                    a,
                    b
                )
                constant_idx += 2

    integral *= 1/poly_down[-1]

    # Ajouter l'intégrale de la partie entière
    integral += integrate_floored_polynomial(floored_poly_up, a, b)
    return integral


if __name__ == "__main__":
    poly_up = [1, 6, 0, -12, 0, 17]
    poly_down = [14, 12, -18]
    a, b = 2, 3
    result = calc_integral(poly_up, poly_down, a, b)

    print("Calcule l'intégrale de:")
    temp = [f'{poly_up[i]}.x^{i}{" + " if i < (len(poly_up) - 1) else ""}' for i in range(len(poly_up))]
    string = ""
    for val in temp:
        string += val
    print(f"{string}")
    print("-" * len(string))
    temp = [f'{poly_down[i]}.x^{i}{" + " if i < (len(poly_down) - 1) else ""}' for i in range(len(poly_down))]
    string = ""
    for val in temp:
        string += val
    print(f"{string}")

    print(f"\nValeur entre {a} et {b}: {result}")
