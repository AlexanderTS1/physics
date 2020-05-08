#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Métodos Numéricos - Práctica 10
    Autor: Alberto García García (agg180@alu.ua.es)

    Notas: Necesario Python 3.7.x.
"""

import matplotlib.pyplot as plt
import numpy as np
import typing


if __name__ == "__main__":

    def punto_fijo_sist(G,p0,tol,maxiter):
        for i in range(maxiter):
            p1 = G(p0)
            if np.linalg.norm(p1-p0)<tol:
                return [p1,i+1]
            p0 = p1
        print('Número máximo de iteraciones alcanzado!')
        return [None,None]

    def newton_sist(F,JF,p0,tol,maxiter):
        for i in range(maxiter):
            y = np.linalg.solve(JF(p0),-F(p0))
            p1 = p0 + y
            if np.linalg.norm(p1-p0)<tol:
                return [p1,i+1]
            p0 = p1
        print('Número máximo de iteraciones alcanzado!')
        return [None,None]

    def JF_approx(F,p0,h):
        n = len(p0)
        JFa = np.zeros((n,n))
        for i in range(n):
            v = np.eye(n)[i]
            JFa[:,i] = (F(p0+h*v)-F(p0))/(h)
        return JFa

    def newton_approx(F,p0,tol,maxiter):
        for i in range(maxiter):
            JFa = JF_approx(F,p0,10**(-2))
            y = np.linalg.solve(JFa,-F(p0))
            p1 = p0 + y
            if np.linalg.norm(p1-p0)<tol:
                return [p1,i+1]
            p0 = p1
        print('Número máximo de iteraciones alcanzado!')
        return [None,None]

    def euler(f,a,b,h,y0):
        N = int((b-a)/h)
        if a+N*h < b:
            N += 1
            h = float(b-a)/N
            print('Se ha recalculado el valor de h =',h)
        t = np.linspace(a,b,N+1)
        y = np.zeros(N+1)
        y[0] = y0
        for i in range(N):
            y[i+1] = y[i] + h*f(t[i],y[i])
        return [t,y]

    def RK4(f,a,b,h,y0):
        N = int((b-a)/h)
        if a+N*h < b:
            N += 1
            h = float(b-a)/N
            print('Se ha recalculado el valor de h =',h)
        t = np.linspace(a,b,N+1)
        y = np.zeros(N+1)
        y[0] = y0
        for i in range(N):
            s1 = h*f(t[i],y[i])
            s2 = h*f(t[i]+h/2,y[i]+s1/2)
            s3 = h*f(t[i]+h/2,y[i]+s2/2)
            s4 = h*f(t[i]+h,y[i]+s3)
            y[i+1] = y[i] + (s1+2*s2+2*s3+s4)/6
        return [t,y]

    # Ejercicio 1 --------------------------------------------------------------
    print("Ejercicio 1 -------------------------------------------------------")

    # a)

    # G(x) = x - A(x) * F(x)
    # x = [x1, x2]
    # F(x) = [f1(x1, x2), f2(x1, x2)]
    # f1(x1, x2) = x1^2 - 2x1 - x2 + 0.5
    # f2(x1, x2) = x1^2 + 4x2^2 - 4

    def _e1_g(p):
        x1, x2 = p
        return np.array(
            [
                0.5 * x1**2 - 0.5 * x2 + 0.25,
                -0.125 * x1**2 -0.5 * x2**2 + x2 + 0.5
            ]
        )

    # b)

    e1_x0_1 = np.array([0.0, 1.0])
    e1_x0_2 = np.array([2.0, 0.0])
    e1_tol = 1e-4
    e1_maxiter = 50

    e1_sol_x0_1 = punto_fijo_sist(_e1_g, e1_x0_1, e1_tol, e1_maxiter)
    e1_sol_x0_2 = punto_fijo_sist(_e1_g, e1_x0_2, e1_tol, e1_maxiter)

    print("Solución (Punto Fijo) para (0, 1) = ", e1_sol_x0_1)
    print("Solución (Punto Fijo) para (2, 0) = ", e1_sol_x0_2)

    # Representación gráfica de ambas funciones y puntos iniciales.
    e1_x, e1_y = np.meshgrid(
        np.arange(-3, 3, 0.1),
        np.arange(-3, 3, 0.1)
    )

    plt.contour(e1_x, e1_y, e1_x**2 - 2.0*e1_x - e1_y + 0.5, [0], colors="blue")
    plt.contour(e1_x, e1_y, e1_x**2 + 4.0*e1_y**2 - 4.0, [0], colors="red")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.show()

    # Como podemos observar en la gráfica, las dos soluciones al sistema se
    # encuentran aproximadamente en (-0.22, 0.99) y (1.90, 0.31).

    # El segundo punto no es capaz de encontrar una solución por hallarse en
    # el límite del dominio de la función y la solución en contra del sentido
    # que marca su derivada.

    # c)

    # Pongamos F(x).
    def _e1_f(p):
        x1, x2 = p

        return np.array(
            [
                x1**2 - 2.0*x1 - x2 +0.5,
                x1**2 + 4.0*x2**2 -4.0
            ]
        )

    # Calculamos el Jacobiano de F(x).
    def _e1_j(p):
        x1, x2 = p

        return np.array(
            [
                [
                    2.0 * x1 - 2.0,
                    -1.0
                ],
                [
                    2.0 * x1,
                    8.0 * x2
                ]
            ]
        )

    # Resolvemos el sistema con el método de Newton implementado.
    e1_soln_x0_1 = newton_sist(_e1_f, _e1_j, e1_x0_1, e1_tol, e1_maxiter)
    e1_soln_x0_2 = newton_sist(_e1_f, _e1_j, e1_x0_2, e1_tol, e1_maxiter)

    print("Solución (Newton) para (0, 1) = ", e1_soln_x0_1)
    print("Solución (Newton) para (2, 0) = ", e1_soln_x0_2)

    # Como podemos comprobar, el método de Newton sí es capaz de obtener las
    # soluciones adecuadas tal y como aproximamos gráficamente en el apartado b).

    # Ejercicio 2 --------------------------------------------------------------
    print("Ejercicio 2 -------------------------------------------------------")

    e2_x = np.array([0.0, 0.6, 0.9, 1.3, 1.6, 2.0, 2.4, 2.7, 3.0])
    e2_y = np.array([1.0, 0.3689, 0.0371, -0.1620, -0.1608, -0.0718, 0.0135, 0.0446, 0.0482])

    def _e2_c(x, alpha, beta):
        return (np.exp(-1.0 * alpha * x) * np.sin(x) +
                np.exp(-1.0 * beta * x) * np.cos(2.0 * x))

    # a)

    e2_tol = 1e-6
    e2_maxiter = 50
    e2_p0 = np.array([0.5, 0.5])

    def _e2_F(p):
        alpha, beta = p
        return np.array(
            [
                -2.0 * np.sum(e2_y - _e2_c(e2_x, alpha, beta)*e2_x * np.exp(-1.0 * alpha * e2_x) * np.sin(e2_x)),
                -2.0 * np.sum(e2_y - _e2_c(e2_x, alpha, beta)*e2_x * np.exp(-1.0 * beta * e2_x) * np.cos(2.0 * e2_x))
            ]
        )

    e2_sol = newton_approx(_e2_F, e2_p0, e2_tol, e2_maxiter)
    print("Solución (Newton Approx.): ", e2_sol)

    # b)

    _e2_c_vect = np.vectorize(_e2_c)
    e2_alphas = np.zeros_like(e2_x)
    e2_alphas = e2_sol[0][0]
    e2_betas = np.zeros_like(e2_x)
    e2_betas = e2_sol[0][1]

    plt.plot(e2_x, e2_y, "green", label="Datos Experimentales")
    plt.plot(e2_x, _e2_c_vect(e2_x, e2_alphas, e2_betas), "red", label="Aproximación")
    plt.xlabel("Tiempo (x)")
    plt.ylabel("Variación de Concentración (y)")
    plt.legend()
    plt.show()

    # Ejercicio 3 --------------------------------------------------------------
    print("Ejercicio 3 -------------------------------------------------------")

    e3_a = 0.0
    e3_b = 2.0
    e3_y0 = 0.5
    e3_h = 0.2

    def _e3_f(t, y):
        return y - t**2 + 1.0

    # a) Punto medio

    def _punto_medio(
        f: typing.Callable,
        a: float,
        b: float,
        h: float,
        y0: float
    ) -> np.array:

        N = int((b-a)/h)

        if a+N*h < b:
            N += 1
            h = float(b-a)/N

        t = np.linspace(a, b, N+1)
        y = np.zeros(N+1)
        y[0] = y0

        for i in range(N):
            r1 = t[i] + h/2.0
            r2 = y[i] + h/2.0 * f(t[i], y[i])
            y[i+1] = y[i] + h * f(r1, r2)

        return [t, y]

    e3_sol_puntomedio = _punto_medio(_e3_f, e3_a, e3_b, e3_h, e3_y0)
    print("Solución (Punto Medio): ", e3_sol_puntomedio)

    # b) Error de truncamiento

    def _e3_analitica(t):
        return (t + 1.0)**2 - 0.5 * np.exp(t)

    e3_sol_analitica = _e3_analitica(e3_sol_puntomedio[0])

    # Representación gráfica de los resultados.
    plt.plot(e3_sol_puntomedio[0], e3_sol_puntomedio[1], "red", label="Punto Medio")
    plt.plot(e3_sol_puntomedio[0], e3_sol_analitica, "green", label="Analitica")
    plt.xlabel("t")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    # Como podemos comprobar, la solución de punto medio es francamente buena.

    # Representamos gráficamente el error.
    e3_error = np.abs(e3_sol_analitica - e3_sol_puntomedio[1])
    plt.plot(e3_sol_puntomedio[0], e3_error, "red", label="Error")
    plt.xlabel("t")
    plt.ylabel("Error Absoluto")
    plt.legend()
    plt.show()

    # Ejercicio 4 --------------------------------------------------------------
    print("Ejercicio 4 -------------------------------------------------------")

    e4_g = 9.81
    e4_m = 5.0
    e4_c = 1.0
    # Comienza en reposo, por lo tanto v0 = 0 [m/s].
    e4_v0 = 0

    # a)

    e4_a = 0.0
    e4_b = 5.0
    e4_h = 1.0

    def _e4_f(t, v):
        return (e4_m * e4_g - e4_c * v) / e4_m

    e4_v_euler = euler(_e4_f, e4_a, e4_b, e4_h, e4_v0)
    print("Solución (Euler) = ", e4_v_euler)
    e4_v_rk4 = RK4(_e4_f, e4_a, e4_b, e4_h, e4_v0)
    print("Solución (RK4) = ", e4_v_rk4)

    # b)

    def _e4_analtica(t):
        return (e4_m * e4_g / e4_c) * (1.0 - np.exp(-1.0 * e4_c * t / e4_m))

    e4_t = np.linspace(e4_a, e4_b, 6)
    e4_v_analitica = _e4_analtica(e4_t)

    # Errores.
    e4_error_euler = np.abs(e4_v_analitica - e4_v_euler[1])
    e4_error_rk4 = np.abs(e4_v_analitica - e4_v_rk4[1])

    print("Error (Euler): ", e4_error_euler)
    print("Error (RK4): ", e4_error_rk4)

    print("RMSE (Euler): ", np.sqrt(np.sum(e4_error_euler**2)/len(e4_error_euler)))
    print("RMSE (RK4): ", np.sqrt(np.sum(e4_error_rk4**2)/len(e4_error_rk4)))

    # El error en cada uno de los puntos es mucho menor en RK4 y el error cuadrático
    # medio también (1.2e-4 comparado con 1.52 de Euler). Por lo tanto, es significativamente
    # mejor el método Runge-Kutta 4.

    # Representación gráfica.

    plt.plot(e4_t, e4_v_euler[1], "red", linestyle="dashed", label="Euler")
    plt.plot(e4_t, e4_v_rk4[1], "blue", linestyle="dashed", label="RK4")
    plt.plot(e4_t, e4_v_analitica, "green", alpha=0.5, label="Analítica")
    plt.xlabel("t")
    plt.ylabel("v")
    plt.legend()
    plt.show()

    # c)

    # Dado que la distancia (posición) es la integral de la velocidad, podemos
    # emplear (por ejemplo) el método del trapecio compuesto para calcular 
    # la integral de la velocidad transcurridos los cinco segundos.

    def trapecio_cerrado(f,a,b):
        return (b-a)*(f(a)+f(b))/2.

    def trapecio_compuesto_cerrado(f,a,b,n):
        xi = np.linspace(a,b,n+1)
        suma = 0.
        for i in range(n):
            suma += trapecio_cerrado(f,xi[i],xi[i+1])
        return suma

    print("Distancia recorrida = {} [m]".format(
        trapecio_compuesto_cerrado(_e4_analtica, 0.0, 5.0, 6)
    ))

    # Dado que hemos calculado anteriormente la velocidad por cada segundo
    # anteriormente y asumiendo que en un tiempo t se recorre un espacio vt.
    # Si asumimos que la velocidad es constante en cada intervalo de tiempo
    # t=1, podemos obtener una aproximación simplemente sumando los valores
    # arrojados por las soluciones de Euler y RK4.

    print("Distancia aproximada: {} [m]".format(np.sum(e4_v_rk4[1][:-1])))

