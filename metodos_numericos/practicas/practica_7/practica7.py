#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad
from scipy.interpolate import pade
import scipy.interpolate as interpol

from practicas.derivint import fourier
from practicas.derivint import fourier_general

if __name__ == "__main__":

    # Ejercicio 1 --------------------------------------------------------------
    print("Ejercicio 1 -------------------------------------------------------")

    e1_f = lambda x: np.sin(x)
    print(fourier(e1_f, np.pi / 2.0, 3))

    # Ejercicio 2 --------------------------------------------------------------
    print("Ejercicio 2 -------------------------------------------------------")

    e2_f = lambda x: x + np.pi

    e2_linspace = np.linspace(-np.pi, np.pi, 128)

    # Vectorizamos la función fourier para representar fácilmente las aproximaciones.
    e2_fourier_vect = np.vectorize(fourier)

    # Representamos la función y las aproximaciones trigonométricas.
    plt.plot(
        e2_linspace,
        e2_f(e2_linspace),
        label="Función"
    )
    for i in range(1,5):
        plt.plot(
            e2_linspace,
            e2_fourier_vect(e2_f, e2_linspace, i),
            label="Orden " + str(i)
        )
    plt.xlabel("Abscisas")
    plt.ylabel("Ordenadas")
    plt.xlim([-np.pi, np.pi])
    plt.legend()
    plt.show()

    # Ejercicio 5 --------------------------------------------------------------
    print("Ejercicio 5 -------------------------------------------------------")

    e5_f = lambda x: np.sin(x)

    # Obtenemos los coeficientes del desarrollo de Taylor.
    e5_taylor = interpol.approximate_taylor_polynomial(e5_f, 0.0, 6, scale=0.1)
    # Potencias crecientes.
    e5_taylor = np.array(e5_taylor)[::-1]
    # Calculamos polinomios de aproximación de Padé.
    e5_p, e5_q = pade(e5_taylor, 3, 3)

    e5_pade = lambda x: e5_p(x) / e5_q(x)
    e5_linspace = np.linspace(-2.0, 2.0, 128)

    plt.plot(
        e5_linspace,
        e5_f(e5_linspace),
        label="Función"
    )
    plt.plot(
        e5_linspace,
        e5_pade(e5_linspace),
        label="Pade (n=m=3)"
    )
    plt.xlabel("Abscisas")
    plt.ylabel("Ordenadas")
    plt.legend()
    plt.show()