import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def pol_lagrange(
        x: np.array,
        y: np.array
) -> np.array:

    vandermonde = np.vander(x)
    return np.matmul(np.linalg.inv(vandermonde), y)


def dif_divididas(
        x: np.array,
        y: np.array
) -> np.array:

    dif_div = np.zeros([len(x), len(x)])
    dif_div[:, 0] = y

    for j in range(1, len(x)):

        for i in range(len(x) - j):

            dif_div[i, j] = ((dif_div[i+1, j-1] - dif_div[i, j-1]) /
                            (x[j+i] - x[i]))

    return dif_div[0]


def pol_newton(
        x: np.array,
        y: np.array,
        grado: int
):

    dif_div = dif_divididas(x, y)

    def _pol_z(z):

        pol = np.zeros(grado + 1)
        pol[0] = dif_div[0]

        for i in range(1, len(pol)):

            g = dif_div[i]

            for j in range(0, i):
                g *= (z - x[j])

            pol[i] = g

        return np.sum(pol)

    return _pol_z


if __name__ == "__main__":

    np.random.seed(42)

    # Ejercicio 1 -----------------------------------------------------------
    print("Ejercicio 1 ****************************************************")

    e1_x_values = np.sort(np.random.uniform(-2.0, 4.0, 7))
    e1_y_values = np.sort(np.random.uniform(-7.0, 5.0, 7))

    e1_pol_newton = pol_newton(e1_x_values, e1_y_values, 6)

    e1_linspace = np.linspace(-2.0, 4.0, 128)
    e1_newton_values = np.zeros(len(e1_linspace))
    for i in range(len(e1_linspace)):
        e1_newton_values[i] = e1_pol_newton(e1_linspace[i])

    #plt.scatter(e1_x_values, e1_y_values, color="red", marker="x", label="Puntos")
    #plt.plot(e1_linspace, e1_newton_values, "--", color="blue", label="Newton")
    #plt.xlabel("Abscisas")
    #plt.ylabel("Ordenadas")
    #plt.xlim([np.min(e1_x_values), np.max(e1_x_values)])
    #plt.legend()
    #plt.show()

    # Ejercicio 2 -----------------------------------------------------------
    print("Ejercicio 2 ****************************************************")

    e2_x_values = np.array([1940, 1950, 1960, 1970, 1980, 1990])
    e2_y_values = np.array([132165, 151326, 179323, 203302, 226542, 249663]) * 1000.0

    e2_pol_lagrange = pol_lagrange(e2_x_values, e2_y_values)

    e2_x_targets = np.array([1930, 1965, 2010])
    e2_y_results = np.polyval(e2_pol_lagrange, e2_x_targets)

    e2_1930_value = 123203000
    e2_1930_error = np.abs(e2_y_results[0] - e2_1930_value)

    print("Población estimada en 1930 {}".format(e2_y_results[0]))
    print("Error = {}".format(e2_1930_error))

    e2_linspace = np.linspace(1930, 2010, 128)

    # plt.scatter(e2_x_values, e2_y_values, color="red", marker="x", label="Puntos")
    # plt.plot(e2_linspace, np.polyval(e2_pol_lagrange, e2_linspace), "--", color="blue", label="Lagrange")
    # plt.xlabel("Abscisas")
    # plt.ylabel("Ordenadas")
    # plt.xlim([1930, 2010])
    # plt.legend()
    # plt.show()

    # El polinomio provoca una serie de curvas que no reflejan los datos dado
    # que posee un grado más elevado del necesario.

    # Ejercicio 3 -----------------------------------------------------------
    print("Ejercicio 3 ----------------------------------------------------")

    e3_x_values = np.array([-3.0, -2.7, -1.8, -0.9, -0.3])
    e3_f = lambda x: np.exp(-2.0 * x) * (x**2 - 3 * x + 2.0)
    e3_y_values = e3_f(e3_x_values)

    e3_pol = np.polyfit(e3_x_values, e3_y_values, 5)

    e3_linspace = np.linspace(-4.0, 0.0, 128)

    # plt.scatter(e3_x_values, e3_y_values, color="red", marker="x", label="Puntos")
    # plt.plot(e3_linspace, np.polyval(e3_pol, e3_linspace), color="green", label="Fit")
    # plt.xlabel("Abscisas")
    # plt.ylabel("Ordenadas")
    # plt.legend()
    # plt.show()

    # Ejercicio 4 -----------------------------------------------------------
    print("Ejercicio 4 ****************************************************")

    def _nodCheb(
            intervalo: np.array,
            n: int
    ) -> np.array:

        nodos_chebysev = np.zeros(n + 1)

        for i in range(n):

            nodos_chebysev[i] = (
                (intervalo[0] + intervalo[1]) / 2.0 -
                (intervalo[1] - intervalo[0]) / 2.0 *
                np.cos(
                    (2.0 * i + 1.0) * np.pi /
                    (2.0 * (n + 1.0))
                )
            )

        return nodos_chebysev

    e4_interval = np.array([1.0, 4.0])
    e4_num_chebysev_nodes = 8
    e4_chebysev_nodes = _nodCheb(e4_interval, e4_num_chebysev_nodes)

    print(e4_chebysev_nodes)

    # Ejercicio 5 -----------------------------------------------------------
    print("Ejercicio 5 ****************************************************")

    e5_f = lambda x: np.exp(-2.0 * x) * (x**2 - 3 * x + 2.0)

    e5_x_values = np.array([-3.0, -2.7, -1.8, -0.9, -0.3])
    e5_y_values = e5_f(e5_x_values)

    e5_x_cheb = _nodCheb(np.array([-4.0, 0.0]), 4)
    e5_y_cheb = e5_f(e5_x_cheb)

    e5_fit = np.polyfit(e5_x_values, e5_y_values, 4)
    e5_cheb = np.polyfit(e5_x_cheb, e5_y_cheb, 4)

    e5_linspace = np.linspace(-4.0, 0.0, 128)

    plt.scatter(e5_x_values, e5_y_values, color="red", marker="x", label="Puntos")
    plt.scatter(e5_x_cheb, e5_y_cheb, color="red", marker="o", label="Puntos Chebyshev")
    plt.plot(e5_linspace, np.polyval(e5_fit, e5_linspace), color="blue", label="Fit")
    plt.plot(e5_linspace, np.polyval(e5_cheb, e5_linspace), color="green", label="Fit Chebyshev")
    plt.plot(e5_linspace, e5_f(e5_linspace), color="yellow", label="Función")
    plt.legend()
    plt.show()
