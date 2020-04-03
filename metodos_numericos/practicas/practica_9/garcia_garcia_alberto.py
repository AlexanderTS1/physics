#! /usr/bin/env python3
# -*- coding: utf-8 -*-

""" Métodos Numéricos - Práctica 9
    Autor: Alberto García García (agg180@alu.ua.es)

    Notas: Necesario Python 3.7.x.
"""

from typing import Tuple

import numpy as np

import scipy.linalg as la

if __name__ == "__main__":

    # Funciones proporcionadas por el script.

    def cambio_filas(A,i,j):
        subs = np.copy(A[i])
        A[i] = A[j]
        A[j] = subs
        return A

    def suma_filas(A,i,j,c):
        subs = np.copy(A[i])
        A[i] = subs + c*A[j]
        return A

    def prod_fila(A,i,c):
        subs = np.copy(A[i])
        A[i] = c*subs
        return A

    def solucionU(A,b):
        n = len(A)
        x = np.zeros(n)
        for i in range(n-1,-1,-1):
            x[i] = (1/A[i,i])*(b[i]-np.sum(A[i]*x))
        return x

    def gauss_parcial(A,b):
        n = len(A) 
        for i in range(n):
            k = np.argmax(abs(A[i::,i])) + i
            cambio_filas(b,i,k) 
            cambio_filas(A,i,k)
            ''' Si además quisiéramos tener unos en la diagonal principal, 
            entonces hemos de activar/quitar # en las siguientes lineas  '''
            # prod_fila(b,i,1/A[i][i])
            # prod_fila(A,i,1/A[i][i])
            for j in range(i+1,n):
                suma_filas(b,j,i,-A[j][i]/A[i][i]) 
                suma_filas(A,j,i,-A[j][i]/A[i][i]) 
        return solucionU(A,b)

    def jacobi(A,b,x0,norma,error,k):
        D = np.diag(np.diag(A))              # diagonal
        L = -np.tril(A-D)                    # triangular inferior
        U = -np.triu(A-D)                    # triangular superior
        M = D
        N = L+U
        B = np.dot(la.inv(M),N)              # M^{-1}N
        c = np.dot(la.inv(M),b)              # M^{-1}b
        val = la.eig(B)[0]                   # Autovalores de B
        ro = max(abs(val)) 
        if ro >= 1: 
            print("El método no es convergente")
            return[x0,0,ro]
        i=1                     # Si es convergente, comenzamos la aproximación.
        while True:
            if i>=k:
                print("El método no converge en",k,"pasos")
                return[x0,k,ro]
            x1 = np.dot(B,x0)+c            # Cada paso: x(m+1) = Bx(m) + c
            if la.norm(x1-x0,norma)<error: # Si el error es menor, salimos del bucle
                return[x1,i,ro]
            i = i+1                        # Si el error no es menor, continuamos
            x0 = x1.copy()                 # Guardamos en x0 el nuevo valor calculado

    # Ejercicio 1 --------------------------------------------------------------
    print("Ejercicio 1 -------------------------------------------------------")

    def solucionL(
        A: np.array,
        b: np.array,
    ) -> np.array:

        n = len(A)
        x = np.zeros(n)
        for i in range(0, n):
            x[i] = (1.0 / A[i,i]) * (b[i] - np.sum(A[i] * x))
        return x

    e1_A = np.array(
        [
            [15.0, 50.0, 67.0],
            [10.0, 20.0, 23.0],
            [5.0, 6.0, 7.0]
        ]
    )
    e1_b = np.array([1.0, 0.0, -1.0])

    # Obtenemos la descomposición LU.
    _, e1_L, e1_U = la.lu(e1_A)
    # Resolvemos Ly = b.
    e1_y = solucionL(e1_L, e1_b)
    # Resolvemos Ux = y.
    e1_x = solucionU(e1_U, e1_y)

    print("Soluciones mediante descomposición LU:")
    print(e1_x)

    # Comprobamos la solución, Ax = b.
    print("Comprobación de solución")
    print(np.matmul(e1_A, e1_x))
    print(e1_b)

    # Ejercicio 2 --------------------------------------------------------------
    print("Ejercicio 2 -------------------------------------------------------")

    def gaussjordan_total(
        A: np.array,
        b: np.array
    ) -> np.array:

        n = len(A)

        for i in range(n):
            # Obtenemos el pivote parcial.
            k = np.argmax(abs(A[i::,i])) + i
            # Colocamos el pivote en la posición de la diagonal adecuada.
            cambio_filas(b,i,k) 
            cambio_filas(A,i,k)
            # Reescalamos la fila para tener 1s en la diagonal principal.
            prod_fila(b, i, 1 / A[i][i])
            prod_fila(A, i, 1 / A[i][i])
            # Hacemos ceros por encima del elemento.
            for j in range(0,i):
                suma_filas(b,j,i,-A[j][i]/A[i][i])
                suma_filas(A,j,i,-A[j][i]/A[i][i])
            # Ceros por encima del elemento actual.
            for j in range(i+1,n):
                suma_filas(b,j,i,-A[j][i]/A[i][i])
                suma_filas(A,j,i,-A[j][i]/A[i][i])

        return b

    e2_A = np.array(
        [
            [1.0, 2.0, -1.0, 3.0],
            [2.0, 0.0, 2.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [3.0, 3.0, -1.0, 2.0]
        ]
    )
    e2_b = np.array([-8.0, 13.0, 8.0, -1.0])

    # Solucionamos.
    e2_gt = gaussjordan_total(e2_A, e2_b)
    print("Solución con Gauss total:")
    print(e2_gt)

    # Comprobación de solución con Gauss parcial.
    e2_gp = gauss_parcial(e2_A, e2_b)
    print("Solución con Gauss parcial:")
    print(e2_gp)

    # Comprobación con linalg.
    e2_linalg = la.solve(e2_A, e2_b)
    print("Solución con linalg:")
    print(e2_linalg)

    # Ejercicio 3 --------------------------------------------------------------
    print("Ejercicio 3 -------------------------------------------------------")

    def inv_parcial(
        A: np.array
    ):
        """ Cálculo de inversa con pivoteo parcial.

            Siguiendo un proceso de eliminación sobre la matriz ampliada [A|I],
            buscamos conseguir el resultado [I|A^-1] por lo que deberemos
            aplicar las transformaciones adecuadas para conseguir una matriz
            identidad en la parte izquierda.

            Nota: en la implementación no utilizamos una matriz ampliada [A|I]
            como tal, sino que replicamos las operaciones sobre dos matrices
            separadas.
        """
        n = len(A)
        A = A.copy()
        A_inv = np.eye(n)

        for i in range(n):
            # Obtenemos el pivote por pivoteo parcial.
            k = np.argmax(abs(A[i::, i])) + i
            # Colocamos el pivote en la posición adecuada de la diagonal.
            cambio_filas(A_inv, i, k)
            cambio_filas(A, i, k)
            # Multiplicación necesaria para tener 1s en la diagonal principal.
            prod_fila(A_inv, i, 1.0 / A[i][i])
            prod_fila(A, i, 1.0 / A[i][i])
            # Conseguimos ceros en todos los elementos encima del actual.
            for j in range(0, i):
              suma_filas(A_inv, j, i, -A[j][i] / A[i][i])
              suma_filas(A, j, i, -A[j][i] / A[i][i])
            # Conseguimos ceros en todos los elementos por debajo del actual.
            for j in range(i+1, n):
              suma_filas(A_inv, j, i, -A[j][i] / A[i][i])
              suma_filas(A, j, i, -A[j][i] / A[i][i])

        return A_inv

    e3_A = np.array(
        [
            [2.0, -1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0, -1.0],
            [-1.0, 0.0, 2.0, -1.0],
            [0.0, -1.0, -1.0, 1.0]
        ]
    )

    # Calculamos la inversa.
    e3_A_inv = inv_parcial(e3_A)
    print("Inversa:")
    print(e3_A_inv)

    # Comprobamos con linalg.
    e3_A_inv_linalg = la.inv(e3_A)
    print("Inversa linalg:")
    print(e3_A_inv)

    # Ejercicio 4 --------------------------------------------------------------
    print("Ejercicio 4 -------------------------------------------------------")

    def gauss_seidel(
        A: np.array,
        b: np.array,
        x: np.array,
        norma: float,
        tolerancia: float,
        k: int
    ) -> Tuple[np.array, int, float]:

        D = np.diag(np.diag(A))
        L = -np.tril(A - D)
        U = -np.triu(A - D)

        M = D - L
        N = U

        M_inv = la.inv(M)
        Mb = np.dot(M_inv, b)
        MN = np.dot(M_inv, N)

        val = la.eig(MN)[0]
        ro = max(abs(val))

        if ro >= 1: 
            print("El método no es convergente")
            return [x, 0, ro]

        i = 0
        while True:

            if i > k:

                print("El método no converge en {} pasos".format(k))
                return [x, k, ro]

            x0 = x.copy()
            x = np.dot(MN, x0) + Mb

            if (la.norm(x - x0, norma) < tolerancia):

                print("El método ha convergido en {} pasos".format(k))
                return [x, i, ro]

            i += 1

    print(gauss_seidel)

    # Ejercicio 5 --------------------------------------------------------------
    print("Ejercicio 5 -------------------------------------------------------")

    # a)
    e5_A = np.array(
        [
            [5.0, 1.0, 2.0],
            [1.0, 4.0, 1.0],
            [2.0, 2.0, 5.0]
        ]
    )
    e5_b = np.array([1.0, 2.0, 2.0])

    # Aproximación inicial, todo unos.
    e5_x0 = np.ones(len(e5_A))

    # Solucionamos con Jacobi.
    e5_a_jacobi = jacobi(e5_A, e5_b, e5_x0, np.inf, 0.0001, 100)
    print("Jacobi:")
    print("x:", e5_a_jacobi[0])
    print("i:", e5_a_jacobi[1])
    print("ro:", e5_a_jacobi[2])

    # Solucionamos con Gauss-Seidel.
    e5_a_gauss = gauss_seidel(e5_A, e5_b, e5_x0, np.inf, 0.0001, 100)
    print("Gauss:")
    print("x:", e5_a_gauss[0])
    print("i:", e5_a_gauss[1])
    print("ro:", e5_a_gauss[2])

    # b)
    e5_A = np.array(
        [
            [1.0, 2.0, -1.0, 3.0],
            [2.0, 0.0, 2.0, -1.0],
            [-1.0, 1.0, 1.0, -1.0],
            [3.0, 3.0, -1.0, 2.0]
        ]
    )
    e5_b = np.array([-8.0, 13.0, 8.0, -1.0])

    # Aproximación inicial, todo ceros.
    e5_x0 = np.zeros(len(e5_A))

    # Solucionamos con Jacobi.
    # e5_a_jacobi = jacobi(e5_A, e5_b, e5_x0, np.inf, 0.0001, 100)
    # No se puede, la matriz M resulta singular y no invertible.

    # Solucionamos con Gauss-Seidel.
    # e5_a_gauss = gauss_seidel(e5_A, e5_b, e5_x0, np.inf, 0.0001, 100)
    # No se puede, la matriz M resulta singular y no invertible.

    # Ejercicio 6 --------------------------------------------------------------
    print("Ejercicio 6 -------------------------------------------------------")

    def e6_matriz(
        n: int
    ):

        A = np.zeros((n, n))

        for i in range(0, n):

            for j in range(0, n):

                if (i <= j):
                    A[i][j] = (i + 1) * (n - j)
                else:
                    A[i][j] = A[j][i]

        return A

    e6_n = 9
    e6_b = np.array(np.arange(1, e6_n+1), dtype=float)
    e6_A = e6_matriz(e6_n)

    print(e6_A)

    # a)

    print("Solución con solve:")
    print(la.solve(e6_A, e6_b))

    # b)

    # x = A^-1 * b
    print("Solución con inversa (parcial y linalg):")
    print(np.matmul(inv_parcial(e6_A), e6_b))
    print(np.matmul(la.inv(e6_A), e6_b))
    # Los números aparentemente parecen estar mal pero son tan pequeños que son
    # cero prácticamente donde deberían ser y por lo tanto debería estar bien.

    # c)

    print("Solución Gauss (parcial y total):")
    print(gauss_parcial(e6_A, e6_b))
    print(gaussjordan_total(e6_A, e6_b))

    # d)

    e6_x0 = np.zeros(len(e6_A))
    
    print("Solución Jacobi:")
    e6_sol_jacobi = jacobi(e6_A, e6_b, e6_x0, np.inf, 1e-5, 100)
    print(e6_sol_jacobi[0])

    print("Solución Gauss-Seidel:")
    e6_sol_gauss = gauss_seidel(e6_A, e6_b, e6_x0, np.inf, 1e-5, 100)
    print(e6_sol_gauss[0])