import numpy as np

from typing import Callable

def nodos_chebyshev(
        intervalo: np.array,
        n: int
) -> np.array:

    nodos = np.zeros(n + 1)

    for i in range(n):

        nodos[i] = (
            (intervalo[0] + intervalo[1]) / 2.0 -
            (intervalo[1] - intervalo[0]) / 2.0 *
            np.cos(
                (2.0 * i + 1.0) * np.pi /
                (2.0 * (n + 1.0))
            )
        )

    return nodos