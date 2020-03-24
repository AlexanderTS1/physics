from typing import Callable

import numpy as np

from scipy.integrate import quad

# Función proporcionada en la solución de la práctica para comparación.
def fourier(f,x,n):
    a0 = (1/np.pi) * quad(lambda x: f(x),-np.pi,np.pi)[0] 
    result = a0/2.
    for k in range(1,n):  
        ak = (1/np.pi) * quad(lambda x: f(x)*np.cos(k*x),-np.pi,np.pi)[0]
        bk = (1/np.pi) * quad(lambda x: f(x)*np.sin(k*x),-np.pi,np.pi)[0]
        result = result + ak*np.cos(k*x) + bk*np.sin(k*x)
    an = (1/np.pi) * quad(lambda x: f(x) * np.cos(n*x),-np.pi,np.pi)[0]        
    return result+ an*np.cos(n*x)


def fourier_general(
    f: Callable,
    l: float,
    n: int,
    x: float
) -> float:
    """
    Fourier general que dada una función f de período 2L sobre el intervalo
    [-L, L), siendo n un número natural y x perteneciente al intervalo,
    devuelve el valor en x de la aproximación trigonométrica de orden n.

    Args:
        f: función a aproximar.
        l: valor del intervalo [-L, L).
        n: orden de la aproximación.
        x: posición en el intervalo para evaluar.

    Returns:
        Valor en x para la aproximación trigonométrica de orden n.
    """

    # Calculamos el coeficiente a_0.
    a0 = (1.0 / l) * quad(lambda x: f(x), -l, l)[0]

    # Calculamos los sumatorios de a_k.
    ak_sum = 0.0

    if (n >= 1):

        ak = np.zeros(n)
        for k in range(1, n+1):
            ak[k-1] = (
                (1.0 / l) * 
                quad(
                    lambda x: f(x) * np.cos(k * np.pi * x / l), -l, l)[0] *
                np.cos(k * np.pi * x / l)
            )

        ak_sum = np.sum(ak)

    # Calculamos los sumatorios de b_k.
    bk_sum = 0.0

    if (n >= 2):

        bk = np.zeros(n-1)
        for k in range(1, n):
            bk[k-1] = (
                (1.0 / l) *
                quad(lambda x: f(x) * np.sin(k * np.pi * x / l), -l, l)[0] *
                np.sin(k * np.pi * x / l)
            )

        bk_sum = np.sum(bk)

    return (a0 + ak_sum + bk_sum)


def dy_tres(
    f: Callable,
    x0: float,
    h: float
) -> float:
    """
    Aproximación de derivada por tres puntos.

    Args:
        f: la función cuya derivada aproximaremos.
        x0: el punto en el que se aproximará.
        h: valor de desplazamiento para los puntos.

    Returns:
        La aproximación de la derivada de la función en un punto empleando la
        fórmula de los tres puntos.
    """

    return (f(x0 + h) - f(x0 - h)) / (2.0 * h)


def dy_cinco(
    f: Callable,
    x0: float,
    h: float
) -> float:
    """
    Aproximación de derivada por cinco puntos.

    Args:
        f: la función cuya derivada aproximaremos.
        x0: el punto en el que se aproximará.
        h: valor de desplazamiento para los puntos.

    Returns:
        La aproximación de la derivada de la función en un punto empleando la
        fórmula de los cinco puntos.
    """

    return (
              f(x0 - 2.0 * h)
              - 8.0 * f(x0 - h)
              + 8.0 * f(x0 + h)
              + f(x0 + 2.0 * h)
           ) / (12.0 * h)


def dy_tres_discreto(
    x: np.array,
    y: np.array
) -> float:
    """
    Aproximación de derivada por tres puntos discreta.

    Args:
        x: tres puntos equiespaciados una distancia h.
        y: valores de la función en dichos tres puntos.

    Returns:
        La aproximación de la derivada de la función en un punto empleando la
        fórmula de los tres puntos.
    """

    h = x[1] - x[0]
    return (y[2] - y[0]) / (2.0 * h)


def dy_cinco_discreto(
    x: np.array,
    y: np.array
) -> float:
    """
    Aproximación de derivada por cinco puntos discreta.

    Args:
        x: cinco puntos equiespaciados una distancia h.
        y: valores de la función en dichos cinco puntos.

    Returns:
        La aproximación de la derivada de la función en un punto empleando la
        fórmula de los cinco puntos.
    """

    h = x[1] - x[0]
    return (y[0] - 8.0 * y[1] + 8.0 * y[3] - y[4]) / (12.0 * h)