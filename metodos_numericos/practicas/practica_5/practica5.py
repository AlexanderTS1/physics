import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quad

from practicas.aproximacion import w_chebyshev
from practicas.aproximacion import w_legendre
from practicas.aproximacion import polinomios_ortogonales_chebyshev
from practicas.aproximacion import polinomios_ortogonales_legendre
from practicas.aproximacion import coeficientes_chebyshev
from practicas.aproximacion import coeficientes_legendre
from practicas.aproximacion import aproximacion_polinomial
from practicas.aproximacion import polinomios_monicos

if __name__ == "__main__":

    np.random.seed(42)

    # Ejercicio 1 -----------------------------------------------------------
    print("Ejercicio 1 ****************************************************")

    def _modelo_discreto_general(
            x: np.array,
            y: np.array,
            m: int
    ) -> np.array :

        n = len(x)

        def _phi(value: float, i: int) -> float:
            return value**i

        a = np.zeros(m + 1)
        phis = np.zeros((m+1, m+1))
        rhs = np.zeros(m + 1)

        for i in range(m + 1):

            for j in range(m + 1):

                acc = 0.0

                for k in range(n):

                    acc += _phi(x[k], j) * _phi(x[k], i)

                phis[i, j] = acc

        for i in range(m + 1):

            acc = 0

            for k in range(n):

                acc += y[k] * _phi(x[k], i)

            rhs[i] = acc

        a = np.matmul(np.linalg.inv(phis), np.transpose(rhs))

        return np.flip(a)

    # Ejercicio 2 -----------------------------------------------------------
    print("Ejercicio 2 ****************************************************")

    e2_x = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    e2_y = np.array([0.000, 0.078, 0.138, 0.192, 0.244])

    e2_pol_1 = _modelo_discreto_general(e2_x, e2_y, 1)
    print(e2_pol_1)
    e2_pol_2 = _modelo_discreto_general(e2_x, e2_y, 2)
    print(e2_pol_2)
    e2_pol_3 = _modelo_discreto_general(e2_x, e2_y, 3)
    print(e2_pol_3)
    e2_pol_4 = _modelo_discreto_general(e2_x, e2_y, 4)
    print(e2_pol_4)

    e2_polyfit_4 = np.polyfit(e2_x, e2_y, 4)
    print(e2_polyfit_4)

    e2_linspace = np.linspace(0.0, 0.4, 128)

    #plt.scatter(e2_x, e2_y, color="red", marker="x", label="Puntos")
    #plt.plot(e2_linspace, np.polyval(e2_pol_1, e2_linspace), label="Grado 1")
    #plt.plot(e2_linspace, np.polyval(e2_pol_2, e2_linspace), label="Grado 2")
    #plt.plot(e2_linspace, np.polyval(e2_pol_3, e2_linspace), label="Grado 3")
    #plt.plot(e2_linspace, np.polyval(e2_pol_4, e2_linspace), label="Grado 4")
    #plt.plot(e2_linspace, np.polyval(e2_polyfit_4, e2_linspace), "--", label="Polyfit 4")
    #plt.legend()
    #plt.show()

    # Ejercicio 3 -----------------------------------------------------------
    print("Ejercicio 3 ****************************************************")

    e3_x = np.linspace(2.0, 4.0, 8)
    e3_f = lambda x: -0.3 * x + 1.4
    e3_y = e3_f(e3_x)

    # a
    e3_min_cuad = _modelo_discreto_general(e3_x, e3_y, 1)

    # c
    e3_x_random = np.random.uniform(2.0, 4.0, 8)
    e3_y_random = e3_f(e3_x_random)
    e3_min_cuad_rand = _modelo_discreto_general(e3_x_random, e3_y_random, 1)

    # d
    e3_y_pert = np.random.uniform(0.0, 1.0, 8) + e3_y_random
    e3_min_cuad_pert = _modelo_discreto_general(e3_x_random, e3_y_pert, 1)

    # b
    e3_linspace = np.linspace(2.0, 4.0, 128)

    #plt.scatter(e3_x, e3_y, color="red", marker="x", label="Puntos")
    #plt.plot(e3_linspace, np.polyval(e3_min_cuad, e3_linspace), color="red", label="Ajuste 1")
    #plt.scatter(e3_x_random, e3_y_random, color="green", marker="x", label="Puntos Random")
    #plt.plot(e3_linspace, np.polyval(e3_min_cuad_rand, e3_linspace), color="green", label="Ajuste Random")
    #plt.scatter(e3_x_random, e3_y_pert, color="blue", marker="x", label="Puntos Perturbados")
    #plt.plot(e3_linspace, np.polyval(e3_min_cuad_pert, e3_linspace), color="blue", label="Ajuste Perturbado")
    #plt.legend()
    #plt.show()

    # Ejercicio 4 -----------------------------------------------------------
    print("Ejercicio 4 ****************************************************")

    e4_x = np.array([0.0, 900.0, 1800.0, 2700.0])
    e4_y = np.array([17.6, 40.4, 67.7, 90.1])

    # a
    e4_regresion = _modelo_discreto_general(e4_x, e4_y, 1)

    # b
    e4_linspace = np.linspace(0.0, 2700.0, 128)

    #plt.scatter(e4_x, e4_y, color="red", marker="x", label="Puntos")
    #plt.plot(e4_linspace, np.polyval(e4_regresion, e4_linspace), label="Regresión")
    #plt.legend()
    #plt.show()

    # c
    # Aproximadamente sí puesto que la relación es lineal.

    # Ejercicio 5 -----------------------------------------------------------
    print("Ejercicio 5 ****************************************************")

    e5_x = np.array([182.0, 232.0, 191.0, 200.0, 148.0, 249.0, 276.0])
    e5_y = np.array([198.0, 210.0, 194.0, 220.0, 138.0, 220.0, 219.0])

    # a
    e5_pol_1 = _modelo_discreto_general(e5_x, e5_y, 1)
    e5_pol_2 = _modelo_discreto_general(e5_x, e5_y, 2)

    # b
    e5_linspace = np.linspace(np.min(e5_x), np.max(e5_x), 128)

    #plt.scatter(e5_x, e5_y, color="red", marker="x", label="Puntos")
    #plt.plot(e5_linspace, np.polyval(e5_pol_1, e5_linspace), label="Grado 1")
    #plt.plot(e5_linspace, np.polyval(e5_pol_2, e5_linspace), label="Grado 2")
    #plt.legend()
    #plt.show()

    # c
    e5_error_1 = np.abs(e5_y - np.polyval(e5_pol_1, e5_x))
    e5_error_2 = np.abs(e5_y - np.polyval(e5_pol_2, e5_x))
    print(e5_error_1)
    print(e5_error_2)

    print("Grado 1 error medio: " + str(np.mean(e5_error_1)))
    print("Grado 2 error medio: " + str(np.mean(e5_error_2)))

    print("Grado 1 error máximo: " + str(np.max(e5_error_1)))
    print("Grado 2 error máximo: " + str(np.max(e5_error_2)))

    # d
    print("Dureza posterior con dureza previa 212: " + str(np.polyval(e5_pol_2, 212.0)))

    # Ejercicio 6 -----------------------------------------------------------
    print("Ejercicio 6 ****************************************************")

    print(polinomios_ortogonales_chebyshev(5))
    print(polinomios_ortogonales_legendre(5))

    # Ejercicio 7 -----------------------------------------------------------
    print("Ejercicio 7 ****************************************************")

    e7_f = lambda x: np.exp(-2.0 * x) * np.cos(3.0 * x)
    e7_grado = 2

    # a

    # Pesos para Chebyshev y Legendre.
    print(w_chebyshev)
    print(w_legendre)

    # Obtenemos polinomios de Chebyshev y los convertimos en mónicos.
    e7_pol_cheb = polinomios_monicos(polinomios_ortogonales_chebyshev(e7_grado))
    print(e7_pol_cheb)

    # Obtenemos polinomios de Legendre y convertimos en mónicos.
    e7_pol_legr = polinomios_monicos(polinomios_ortogonales_legendre(e7_grado))
    print(e7_pol_legr)

    # Calculamos coeficientes para Chebyshev y Legendre
    e7_a_cheb = coeficientes_chebyshev(e7_f, e7_pol_cheb, 2)
    print("Coeficientes Chebyshev: " + str(e7_a_cheb))

    e7_a_legr = coeficientes_legendre(e7_f, e7_pol_legr, 2)
    print("Coeficientes Legendre: " + str(e7_a_legr))

    # b

    e7_linspace = np.linspace(-1.0, 1.0, 128)

    plt.plot(e7_linspace, e7_f(e7_linspace), "red", label="Función")
    plt.plot(e7_linspace, aproximacion_polinomial(e7_a_cheb, e7_pol_cheb, e7_linspace), "blue", label="Chebyshev")
    plt.plot(e7_linspace, aproximacion_polinomial(e7_a_legr, e7_pol_legr, e7_linspace), "green", label="Legendre")
    plt.xlabel("Abscisas")
    plt.ylabel("Ordenadas")
    plt.show()

    # c

    e7_norm_cheb = quad(
        lambda x: w_chebyshev(x) * (e7_f(x) - aproximacion_polinomial(e7_a_cheb, e7_pol_cheb, x))**2,
        -1.0,
        1.0
    )[0]**0.5

    print("Norma Chebyshev: " + str(e7_norm_cheb))

    e7_norm_legr = quad(
        lambda x: w_legendre(x) * (e7_f(x) - aproximacion_polinomial(e7_a_legr, e7_pol_legr, x))**2,
        -1.0,
        1.0
    )[0]**0.5

    print("Norma Legendre: " + str(e7_norm_legr))