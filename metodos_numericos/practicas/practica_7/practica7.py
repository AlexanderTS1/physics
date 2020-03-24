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
from practicas.derivint import dy_tres
from practicas.derivint import dy_cinco
from practicas.derivint import dy_tres_discreto
from practicas.derivint import dy_cinco_discreto

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

    # Ejercicio 4 --------------------------------------------------------------
    print("Ejercicio 3 -------------------------------------------------------")

    def e3_f(
        x,
        T
    ):
        if -T/2<= x < -T/4:
            return -1.
        elif -T/4 <= x < T/4:
            return 1.
        else:
            return -1.

    e3_f_vect = np.vectorize(e3_f) 
    e3_T = 2.0

    e3_linspace = np.linspace(-e3_T/2., e3_T/2., 128)
    plt.plot(e3_linspace, e3_f_vect(e3_linspace, e3_T),'k')
    plt.show()

    # Tengamos en cuenta que si f(x) es periódica de periodo T=2L, la función
    # g(t)=f(L*t/Pi) es periódica de periodo 2Pi y podemos aplicar las fórmulas 
    # de la presentación. En este caso, el periodo es T=2, de donde L=T/2=1.

    e3_L = e3_T / 2.0

    # Paso a paso: 

    e3_a0 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T)),-e3_L,e3_L)[0]
    e3_a1 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.cos(1*np.pi*x/e3_L)),-e3_L,e3_L)[0]
    e3_b1 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.sin(1*np.pi*x/e3_L)),-e3_L,e3_L)[0]
    e3_a2 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.cos(2*np.pi*x/e3_L)),-e3_L,e3_L)[0]
    e3_b2 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.sin(2*np.pi*x/e3_L)),-e3_L,e3_L)[0]
    e3_a3 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.cos(3*np.pi*x/e3_L)),-e3_L,e3_L)[0]
    e3_b3 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.sin(3*np.pi*x/e3_L)),-e3_L,e3_L)[0]  
    e3_a4 = (1/e3_L) * quad(lambda x: (e3_f_vect(x,e3_T) * np.cos(4*np.pi*x/e3_L)),-e3_L,e3_L)[0]

    # Para construir la aproximación final, deshacemos el cambio teniendo en cuenta que
    # f(x)=g(Pi*x/L).

    def e3_approx(x,L):
        return e3_a0/2. + (
            e3_a1*np.cos(np.pi*x/e3_L)
            + e3_b1*np.sin(np.pi*x/e3_L)
        ) + (
            e3_a2*np.cos(2*np.pi*x/e3_L)
            + e3_b2*np.sin(2*np.pi*x/e3_L)
        ) + (
            e3_a3*np.cos(3*np.pi*x/e3_L)
            + e3_b3*np.sin(3*np.pi*x/e3_L)
        ) + e3_a4*np.cos(4*np.pi*x/e3_L)

    e3_linspace_2 = np.linspace(-e3_L, e3_L, 128)

    plt.plot(e3_linspace_2, e3_f_vect(e3_linspace_2, e3_T),'k', label="Función")
    plt.plot(e3_linspace_2, e3_approx(e3_linspace_2, e3_L),'b', label="Aproximación")
    plt.legend()
    plt.xlabel("Abscisas")
    plt.ylabel("Ordenadas")
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

    # Ejercicio 6 --------------------------------------------------------------
    print("Ejercicio 6 -------------------------------------------------------")

    print(dy_tres)
    print(dy_cinco)

    # Ejercicio 7 --------------------------------------------------------------
    print("Ejercicio 7 -------------------------------------------------------")

    e7_f = lambda x: np.exp(-x)
    e7_df = lambda x: -np.exp(-x)
    e7_x0 = 2.0

    print("Aproximación de tres puntos h=0.1 en x0={} es {} ".format
        (
            e7_x0,
            dy_tres(e7_f, e7_x0, 0.1)
        )
    )

    print("Aproximación de cinco puntos h=0.1 en x0={} es {} ".format
        (
            e7_x0,
            dy_cinco(e7_f, e7_x0, 0.1)
        )
    )

    print("Aproximación de tres puntos h=0.05 en x0={} es {} ".format
        (
            e7_x0,
            dy_tres(e7_f, e7_x0, 0.05)
        )
    )

    print("Aproximación de cinco puntos h=0.05 en x0={} es {} ".format
        (
            e7_x0,
            dy_cinco(e7_f, e7_x0, 0.05)
        )
    )

    print("Valor real de df en x0={} es {} ".format
        (
            e7_x0,
            e7_df(e7_x0)
        )
    )

    # Ejercicio 8 --------------------------------------------------------------
    print("Ejercicio 8 -------------------------------------------------------")

    print(dy_tres_discreto)
    print(dy_cinco_discreto)

    # Ejercicio 9 --------------------------------------------------------------
    print("Ejercicio 9 -------------------------------------------------------")

    e9_x = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0])
    e9_y = np.array([0.0, 20.0, 50.0, 100.0, 150.0, 180.0, 200.0, 210.0, 240.0, 280.0, 335.0, 400.0, 465.0])

    # a)

    plt.plot(e9_x, e9_y, marker="x", label="Datos")
    plt.xlabel("Tiempo [s]")
    plt.ylabel("Espacio [m]")
    plt.show()

    # b)

    print("Aproximaciones de tres puntos para h=5")

    for i in range(2, len(e9_x)-2):
        v = dy_tres_discreto(
            e9_x[np.array([i-1,i,i+1])],
            e9_y[np.array([i-1,i,i+1])]
        )
        print("Tiempo = " + str(e9_x[i]))
        print("Velocidad = " + str(v))

    print("Aproximaciones de tres puntos para h=10")

    for i in range(2, len(e9_x)-2):
        v = dy_tres_discreto(
            e9_x[np.array([i-2,i,i+2])],
            e9_y[np.array([i-2,i,i+2])]
        )
        print("Tiempo = " + str(e9_x[i]))
        print("Velocidad = " + str(v))

    # c) TODO.

    # d) TODO.