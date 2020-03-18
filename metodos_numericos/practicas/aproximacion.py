import numpy as np

from typing import Callable
from scipy.integrate import quad

w_legendre = lambda x: 1.0
w_chebyshev = lambda x: 1.0 / np.sqrt(1.0 - x**2.0)

def polinomios_ortogonales_chebyshev(
        n: int
) -> np.array:

    def _chebyshev(
            i: int
    ) -> np.array:

        if i == 0:
            return np.array([1.0])
        elif i == 1:
            return np.array([1.0, 0.0])
        else:
            tn1 = _chebyshev(i-1)
            tn2 = _chebyshev(i-2)
            tn = np.polysub(np.polymul(np.array([2.0, 0.0]), tn1), tn2)
            return tn

    t = []

    for i in range(n+1):

        t.append(_chebyshev(i))

    return np.array(t)


def polinomios_ortogonales_legendre(
        n: int
) -> np.array:

    def _legendre(
            i: int
    ) -> np.array:

        if i == 0:
            return np.array([1.0])
        elif i == 1:
            return np.array([1.0, 0.0])
        else:
            tn1 = _legendre(i-1)
            tn2 = _legendre(i-2)
            tn = np.polysub(
                np.polymul(
                    [(2.0 * i + 1.0) / (i + 1.0), 0.0],
                    tn1
                ),
                np.polymul(
                    [i / (i + 1.0)],
                    tn2
                )
            )
            
            return tn

    t = []

    for i in range(n + 1):

        t.append(_legendre(i))

    return np.array(t)


def polinomios_monicos(
        polinomios: np.array
) -> np.array:

    for i in range(len(polinomios)):
        polinomios[i] /= polinomios[i][0]

    return polinomios


def coeficientes_chebyshev(
        funcion: Callable,
        polinomios: np.array,
        grado: int
) -> np.array:

    coeffs = np.zeros(grado + 1)
    for i in range(len(coeffs)):
        coeffs[i] = (
            quad(lambda x: w_chebyshev(x) * funcion(x) * np.polyval(polinomios[i], x), -1.0, 1.0)[0] /
            quad(lambda x: w_chebyshev(x) * np.polyval(polinomios[i], x)**2, -1.0, 1.0)[0]
        )

    return coeffs

def coeficientes_legendre(
        funcion: Callable,
        polinomios: np.array,
        grado: int
) -> np.array:

    coeffs = np.zeros(grado + 1)
    for i in range(len(coeffs)):
        coeffs[i] = (
            quad(lambda x: w_legendre(x) * funcion(x) * np.polyval(polinomios[i], x), -1.0, 1.0)[0] /
            quad(lambda x: w_legendre(x) * np.polyval(polinomios[i], x)**2, -1.0, 1.0)[0]
        )

    return coeffs

def aproximacion_polinomial(
      coeffs: np.array,
      polys: np.array,
      x: float
) -> float:

    approx = 0.0

    for i in range(len(coeffs)):

        approx += coeffs[i] * np.polyval(polys[i], x)

    return approx

