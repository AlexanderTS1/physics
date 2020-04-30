#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Métodos Numéricos - Práctica 10
    Autor: Alberto García García (agg180@alu.ua.es)

    Notas: Necesario Python 3.7.x.
"""

import numpy as np
import typing


if __name__ == "__main__":

    # Funciones proporcionadas por el script.

    def busqueda_incremental(f,a,b,n):
        # f: funcion que determina la ecuación
        # a: extremo inferior del intervalo
        # b: extremo superior del intervalo
        # n: número de subintervalos
        extremos=np.linspace(a,b,n+1)
        intervalos=np.zeros((n,2))
        lista=[]
        for i in range(n):
            intervalos[i,0]=extremos[i]
            intervalos[i,1]=extremos[i+1]
            if f(extremos[i])*f(extremos[i+1])<=0:
                lista.append(i)
        return intervalos[lista,::]

    def biseccion(f,a,b,tol):
        # f: funcion que determina la ecuación
        # a: extremo inferior del intervalo
        # b: extremo superior del intervalo
        # tol: tolerancia
        i = 0
        while abs(b-a)>=tol:
            p = (a+b)/2.0
            if f(p) == 0:
                return [p,i]
            else:
                if f(a)*f(p)>0:
                    a = p
                else:
                    b = p
            i = i+1
        return [p,i]

    def punto_fijo(g,p0,tol,maxiter):
        # g: funcion que determina la ecuación
        # p0: punto inicial
        # tol: tolerancia 
        # maxiter: máximo número de iteraciones permitidas
        for i in range(maxiter):      
            p1 = g(p0)
            if abs(p1-p0)<tol:
                return [p1,i]
            p0 = p1
        print('Número máximo de iteraciones alcanzado!')
        return [None,None]

    def newton(f,df,p0,tol,maxiter):
        # f: funcion que determina la ecuación
        # df: derivada de f
        # p0: punto inicial
        # tol: tolerancia 
        # maxiter: máximo número de iteraciones permitidas
        for i in range(maxiter):      
            p1 = p0-f(p0)/df(p0)
            if abs(p1-p0)<tol:
                return [p1,i]
            p0 = p1
        print('Número máximo de iteraciones alcanzado!')
        return [None,None]

    def regula_falsi(f,p0,p1,tol,maxiter):
        # f: funcion que determina la ecuación
        # p0 y p1: puntos iniciales
        # tol: tolerancia 
        # maxiter: máximo número de iteraciones permitidas
        p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0)) #Primer paso
        i = 1
        while  (i<maxiter) and abs(p2-p1)>=tol:
            i = i+1
            if f(p1)*f(p2)<0:
                p0 = p1
            p1 = p2
            p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0)) 
        return [p2 ,i]

    # Ejercicio 1 --------------------------------------------------------------
    print("Ejercicio 1 -------------------------------------------------------")

    e1_f = lambda x: np.sin(x) - np.log(x)
    e1_int = [2.0, 2.5]
    e1_eps = 1e-4

    def _busca_solucion_incremental(
        f: typing.Callable,
        a: float,
        b: float,
        n: int,
        eps: float
    ) -> list:

        """ Cálculo de solución mediante búsqueda incremental recursiva.

        Esta función se aprovecha de la función "busqueda_incremental" ya
        implementada exclusivamente para obtener las raíces de una función.

        El procedimiento consiste en utilizar la función "busqueda_incremental"
        para obtener una serie de intervalos candidatos en los que se cumple
        el Teorema de Bolzano y por lo tanto existe una raíz f(x) = 0.

        Esos intervalos serán a su vez descompuestos en otros intervalos más
        reducidos mediante llamadas recursivas a esta propia función.

        Existen dos posibles casos base o de parada para esta recursividad:

        * El intervalo no cumple el Teorema de Bolzano, por lo tanto no existe
          raíz f(x) = 0 y se devuelve una solución vacía.
        * El intervalo cumple el Teorema de Bolzano y además el tamaño del mismo
          se encuentra por debajo de un épsilon especificado. En este caso
          aproximaremos la raíz como el punto medio en dicho intervalo.

        En cualquier otro caso (el intervalo cumple el Teorema de Bolzano pero
        su tamaño está por encima del épsilon especificado), se descompondrá
        el mismo en subintervalos y se procederá con una llamada recursiva en
        cada uno de ellos.

        Las sucesivas llamadas que resulten en una solución irán acumulando en
        una lista las soluciones encontradas (empleando extend en lugar de
        append para no acabar con una lista de listas infinita).

        La precisión de las soluciones dependerá del tamaño máximo de intervalo
        que escojamos.

        Argumentos:

            f: función que determina la ecuación.
            a: extremo inferior del intervalo.
            b: extremo superior del intervalo.
            n: número de subintervalos a crear.
            eps: tamaño máximo de intervalo.

        Returns:

            Lista de raíces.

        """

        if (f(a)*f(b) > 0):
            return []
        if (f(a)*f(b) <= 0) and (abs(b-a) <= eps):
            return [(a+b)/2.0]
        else:

            soluciones = []

            intervalos_busqueda = busqueda_incremental(f, a, b, n)
            for intervalo in intervalos_busqueda:
                soluciones.extend(_busca_solucion_incremental(
                    f, intervalo[0], intervalo[1], 4, eps
                ))

            return soluciones

    # Comprobamos que la función arroja la solución correcta para la función
    # dada en el ejercicio sin(x) = log(x), la cual tiene una única ráiz en
    # x=2.21910714891375...
    #
    # Con un épsilon de 1e-4 deberíamos obtener una solución aproximada de
    # x=2.2191...
    e1_sols = _busca_solucion_incremental(e1_f, e1_int[0], e1_int[1], 4, e1_eps)
    print("Raíz de la función mediante búsqueda incremental recursiva: ")
    print(e1_sols)

    # Ejercicio 2 --------------------------------------------------------------
    print("Ejercicio 2 -------------------------------------------------------")

    # Debemos encontrar la raíz de la función cos(x) - x * e^x = 0.
    e2_f = lambda x: np.cos(x) - x * np.exp(x)
    # Para aplicar los métodos de punto fijo necesitamos una transformación
    # g(x) = x. En nuestro caso es muy sencillo ya que si cos(x) - x * e^x = 0
    # entonces x = cos(x) / e^x y por lo tanto g(x) = cos(x) / e^x.
    e2_g = lambda x: np.cos(x) / np.exp(x)
    e2_tol = 1e-4
    e2_maxiter = 50

    def aitken_puntofijo(
        g: typing.Callable,
        p0: float,
        tol: float,
        maxiter: int
    ) :

        """
        Método de punto fijo con aceleración de convergencia de Aitken.

        Args:
            g: función tal que g(x) = x
            p0: punto inicial.
            tol: tolerancia del método.
            maxiter: número máximo de iteraciones.

        Returns:

            None si no converge y una tupla [solución, iteraciones] si
            se cumple el criterio de parada.
        """

        p1 = g(p0)
        p2 = g(p1)

        p0_star = p0 - ((p1 - p0)**2 / (p2 - 2.0 * p1 + p0))

        for i in range(maxiter):

            p3 = g(p2)
            p1_star = p1 - (((p2 - p1)**2.0) / (p3 - 2.0 * p2 + p1))

            if (abs(p0_star - p1_star) < tol):
                return [p1_star, i]

            p0_star = p1_star
            p1 = p2
            p2 = p3

        print("Máximo número de iteraciones alcanzado sin convergencia...")
        return [None, None]

    e2_sol_puntofijo = punto_fijo(e2_g, 2.0, e2_tol, e2_maxiter)
    print("Solución de punto fijo: ", e2_sol_puntofijo)
    e2_sol_aitken = aitken_puntofijo(e2_g, 2.0, e2_tol, e2_maxiter)
    print("Solución con aceleración de Aitken: ", e2_sol_aitken)

    # Podemos comprobar que, mientras que la solución de punto fijo necesita
    # 46 iteraciones para converger en una solución con la tolerancia dada,
    # el método con aceleración de Aitken converge en solamente 13 iteraciones.

    # Ejercicio 3 --------------------------------------------------------------
    print("Ejercicio 3 -------------------------------------------------------")

    e3_f = lambda x: np.cos(x) - x * np.exp(x)
    e3_int = [0.0, 2.0]
    e3_h = 1e-2
    e3_tol = 1e-4
    e3_maxiter = 50

    def dy_cinco(
        f: typing.Callable,
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

    def _newton_5ptos(
        f: typing.Callable,
        h: float,
        p0: float,
        tol: float,
        maxiter: int
    ) -> list:

        """
        Método de Newton para ecuaciones no lineales empleando la aproximación
        de la derivada de la función por cinco puntos en lugar de la derivada
        analítica de la misma.

        Args:
            f: la función cuya raíz obtendremos.
            h: valor de desplazamiento para los puntos de la aprox. de derivada.
            h0: el punto de inicio del método.
            tol: tolerancia del método.
            maxiter: número máximo de iteraciones permitidas.

        Returns:

            Raíz de la función especificada.
        """

        for i in range(maxiter):
            # Empleamos la apromixación de la derivada por cinco puntos en lugar
            # de la derivada analítica.
            p1 = p0 - f(p0) / dy_cinco(f, p0, h)

            if abs(p1 - p0) < tol:
                return [p1, i]

            p0 = p1

        print("Número máximo de iteraciones alcanzado...")
        return [None, None]

    e3_intervalo = busqueda_incremental(e3_f, e3_int[0], e3_int[1], 4)
    print("El intervalo para escoger el punto inicial es: ")
    print(e3_intervalo)
    e3_p0 = (e3_intervalo[0][0] + e3_intervalo[0][1]) / 2.0
    print("El punto inicial es el punto medio de dicho intervalo: ")
    print(e3_p0)
    e3_sol = _newton_5ptos(e3_f, e3_h, e3_p0, e3_tol, e3_maxiter)
    print("La raíz se encuentra en: ")
    print(e3_sol)

    # Ejercicio 4 --------------------------------------------------------------
    print("Ejercicio 4 -------------------------------------------------------")

    e4_f = lambda x: np.cos(x) - x * np.exp(x)
    e4_int = [0.0, 2.0]
    e4_tol = 1e-4
    e4_maxiter = 50


    def secante_parada(
        f,
        p0,
        p1,
        tol,
        maxiter,
        criterio = 1
    ):
        """
        Método de la secante con selección de criterio de parada.

        Args:
            f: función cuyas raíces necesitamos.
            p0: extremo inferior.
            p1: extremo superior.
            tol: tolerancia del método.
            maxiter: número máximo de iteraciones.
            criterio: criterio de parada (1), (2) ó (3).

        Returns:

            Tupla [solución, iteraciones].
        """

        def _parada_1(f, p1, p2, tol):
            return (abs(p2 - p1) < tol)
        def _parada_2(f, p1, p2, tol):
            return ((abs(p2 - p1) / p2) < tol)
        def _parada_3(f, p1, p2, tol):
            return (f(p2) < tol)

        parada_funs = [_parada_1, _parada_2, _parada_3]
        parada = parada_funs[criterio-1]

        p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0))
        i = 1
        while (i<maxiter) and not parada(f, p1, p2, tol): 
            i=i+1
            p0 = p1
            p1 = p2
            p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0))
        return [p2,i]

    e4_intervalo = busqueda_incremental(e4_f, e4_int[0], e4_int[1], 8)
    print("Puntos iniciales: ", e4_intervalo[0])

    e4_sol_1 = secante_parada(
        e4_f, e4_intervalo[0][0],
        e4_intervalo[0][1],
        e4_tol,
        e4_maxiter,
        1
    )
    print("Solución con criterio (1): ", e4_sol_1)
    e4_sol_2 = secante_parada(
        e4_f, e4_intervalo[0][0],
        e4_intervalo[0][1],
        e4_tol,
        e4_maxiter,
        2
    )
    print("Solución con criterio (2): ", e4_sol_2)
    e4_sol_3 = secante_parada(
        e4_f,
        e4_intervalo[0][0],
        e4_intervalo[0][1],
        e4_tol,
        e4_maxiter,
        3
    )
    print("Solución con criterio (3): ", e4_sol_3)

    # Ejercicio 5 --------------------------------------------------------------
    print("Ejercicio 5 -------------------------------------------------------")

    # Encontrar la raíz quinta de dos se puede expresar también como encontrar
    # la raíz de la ecuación x - 2^(1/5) = 0 que podemos obtener mediante los
    # métodos estudiados.
    e5_f = lambda x: x - 2**(1.0 / 5.0)
    e5_tol = 1e-4
    e5_iter = 50

    # La elección del intervalo para todos estos métodos está clara: la raíz
    # debe ser mayor que 1.0 (de lo contrario sería imposible que al elevarse
    # a la quinta potencia incrementara su valor hasta converger en 2.0) y por
    # lógica además debe ser menor que 2.0 (ya que de lo contrario al elevarse
    # a la quinta resultaría en un número mayor que 2.0).
    #
    # Partiendo de ese intervalo [1.0, 2.0] emplearemos la función búsqueda
    # incremental para refinar todavía más el intervalo. En el caso de los
    # métodos que requieran un punto, usaremos el punto medio.

    e5_int = busqueda_incremental(e5_f, 1.0, 2.0, 4)
    e5_p0 = e5_int[0][0]
    e5_p1 = e5_int[0][1]

    e5_biseccion = biseccion(e5_f, e5_p0, e5_p1, e5_tol)
    print("Aproximación con bisección: ", e5_biseccion)

    e5_newton = newton(e5_f, lambda x: 1.0, (e5_p0 + e5_p1) / 2.0, e5_tol, e5_iter)
    print("Aproximación con Newton: ", e5_newton)

    e5_secante = secante_parada(e5_f, e5_p0, e5_p1, e5_tol, e5_iter, 1)
    print("Aproximación con Secante con parada(1): ", e5_secante)

    e5_regula = regula_falsi(e5_f, e5_p0, e5_p1, e5_tol, 50)
    print("Aproximación con Regula-Falsi: ", e5_regula)
