import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.interpolate as si


if __name__ == "__main__":

    np.random.seed(42)

    # Ejercicio 1 -----------------------------------------------------------
    print("Ejercicio 1 ****************************************************")

    e1_x = np.array([1.0, 2.0, 3.0, 4.0])
    e1_y = np.array([0.5, 4.0, 2.0, 0.7])

    e1_spline_3_natural = si.CubicSpline(e1_x, e1_y, bc_type="natural")
    e1_spline_3_frontera = si.CubicSpline(e1_x, e1_y, bc_type=((1, 0.3), (2, -0.3)))

    e1_linspace = np.linspace(1.0, 4.0, 128)

    #plt.scatter(e1_x, e1_y, color="red", marker="x", label="Puntos")
    #plt.plot(e1_linspace, e1_spline_3_natural(e1_linspace), label="Natural")
    #plt.plot(e1_linspace, e1_spline_3_frontera(e1_linspace), label="Frontera")
    #plt.legend()
    #plt.show()

    # Ejercicio 2 -----------------------------------------------------------
    print("Ejercicio 2 ****************************************************")

    e2_f = lambda x: 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-1.0 * x**2 / 2.0)

    e2_x = np.linspace(-3.0, 3.0, 5)
    e2_y = e2_f(e2_x)

    e2_poly = np.polyfit(e2_x, e2_y, 4)
    e2_spline_3_natural = si.CubicSpline(e2_x, e2_y, bc_type="natural")

    e2_linspace = np.linspace(-3.0, 3.0, 128)

    e2_range = np.arange(-3.0, 3.0, 0.2)
    e2_f_value = e2_f(e2_range)
    e2_poly_value = np.polyval(e2_poly, e2_range)
    e2_spline_value = e2_spline_3_natural(e2_range)

    e2_poly_error = np.abs(e2_f_value - e2_poly_value)
    e2_spline_error = np.abs(e2_f_value - e2_spline_value)

    e2_table = pd.DataFrame()
    e2_table["X"] = e2_range
    e2_table["Polyfit Error"] = e2_poly_error
    e2_table["Spline Error"] = e2_spline_error

    #print(e2_table)
#
    #plt.scatter(e2_x, e2_y, color="red", marker="x", label="Puntos")
    #plt.plot(e2_linspace, np.polyval(e2_poly, e2_linspace), label="Polyfit")
    #plt.plot(e2_linspace, e2_spline_3_natural(e2_linspace), label="Spline")
    #plt.plot(e2_linspace, e2_f(e2_linspace), label="Function")
    #plt.plot(e2_range, e2_poly_error, "--", label="Polyfit Error")
    #plt.plot(e2_range, e2_spline_error, "--", label="Spline Error")
    #plt.legend()
    #plt.show()

    # Ejercicio 3 -----------------------------------------------------------
    print("Ejercicio 3 ****************************************************")

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

    e3_f = lambda x: 1.0 / (1.0 + x**2)

    e3_x_6 = np.linspace(-5.0, 5.0, 6)
    e3_x_8 = np.linspace(-5.0, 5.0, 8)
    e3_x_10 = np.linspace(-5.0, 5.0, 10)
    e3_x_12 = np.linspace(-5.0, 5.0, 12)

    e3_poly_5 = np.polyfit(e3_x_6, e3_f(e3_x_6), 5)
    e3_poly_7 = np.polyfit(e3_x_8, e3_f(e3_x_8), 7)
    e3_poly_9 = np.polyfit(e3_x_10, e3_f(e3_x_10), 9)
    e3_poly_11 = np.polyfit(e3_x_12, e3_f(e3_x_12), 11)

    e3_x_cheb = _nodCheb(np.array([-5.0, 5.0]), 6)
    e3_poly_5_cheb = np.polyfit(e3_x_cheb, e3_f(e3_x_cheb), 5)

    e3_x_spline = np.linspace(-5.0, 5.0, 9)
    e3_y_spline = e3_f(e3_x_spline)
    e3_spline_3_frontera = si.CubicSpline(e3_x_spline, e3_y_spline, bc_type=((1, 0.014), (2, -0.014)))

    e3_linspace = np.linspace(-5.0, 5.0, 128)

    #plt.plot(e3_linspace, e3_f(e3_linspace), "--", label="Función")
    #plt.plot(e3_linspace, np.polyval(e3_poly_5, e3_linspace), label="Grado 5")
    #plt.plot(e3_linspace, np.polyval(e3_poly_7, e3_linspace), label="Grado 7")
    #plt.plot(e3_linspace, np.polyval(e3_poly_9, e3_linspace), label="Grado 9")
    #plt.plot(e3_linspace, np.polyval(e3_poly_11, e3_linspace), label="Grado 11")
    #plt.plot(e3_linspace, np.polyval(e3_poly_5_cheb, e3_linspace), label="Grado 5 Chebyshev")
    #plt.plot(e3_linspace, e3_spline_3_frontera(e3_linspace), label="Spline")
    #plt.legend()
    #plt.show()

    # Ejercicio 4 -----------------------------------------------------------

    e4_x = np.array([5.0, 10.0, 15.0, 20.0, 25.0])
    e4_y = np.array([49.0, 105.0, 172.0, 253.0, 352.0])

    e4_poly_2 = np.polyfit(e4_x[::2], e4_y[::2], 2)
    print("Poly_2 para Kp=13 es {}".format(np.polyval(e4_poly_2, 13)))

    e4_poly_4 = np.polyfit(e4_x, e4_y, 4)
    print("Poly_4 para Kp=7 es {}".format(np.polyval(e4_poly_4, 7)))
    print("Poly_4 para Kp=12 es {}".format(np.polyval(e4_poly_4, 12)))
    print("Poly_4 para Kp=17 es {}".format(np.polyval(e4_poly_4, 17)))
    print("Poly_4 para Kp=22 es {}".format(np.polyval(e4_poly_4, 22)))

    e4_spline_3 = si.CubicSpline(e4_x, e4_y, bc_type="natural")
    print("Spline_3 para Kp=7 es {}".format(e4_spline_3(7)))
    print("Spline_3 para Kp=12 es {}".format(e4_spline_3(12)))
    print("Spline_3 para Kp=17 es {}".format(e4_spline_3(17)))
    print("Spline_3 para Kp=22 es {}".format(e4_spline_3(22)))

    e4_linspace = np.linspace(5.0, 25.0, 128)

    #plt.scatter(e4_x, e4_y, color="red", marker="x", label="Puntos")
    #plt.plot(e4_linspace, np.polyval(e4_poly_2, e4_linspace), label="Poly2")
    #plt.plot(e4_linspace, np.polyval(e4_poly_4, e4_linspace), label="Poly4")
    #plt.plot(e4_linspace, e4_spline_3(e4_linspace), label="Spline")
    #plt.legend()
    #plt.show()

    # Ejercicio 5 ------------------------------------------------------------

    e5_f = lambda t: 100.0 / (2.0 + 999.0 * np.exp(-2.1 * t))

    e5_x_5 = np.linspace(0.0, 7.0, 5)
    e5_x_7 = np.linspace(0.0, 7.0, 7)
    e5_x_9 = np.linspace(0.0, 7.0, 9)

    e5_poly_4 = np.polyfit(e5_x_5, e5_f(e5_x_5), 4)
    e5_poly_6 = np.polyfit(e5_x_7, e5_f(e5_x_7), 6)
    e5_poly_8 = np.polyfit(e5_x_9, e5_f(e5_x_9), 8)

    e5_x_5_cheb = _nodCheb(np.array([0.0, 7.0]), 4)
    e5_x_7_cheb = _nodCheb(np.array([0.0, 7.0]), 6)
    e5_x_9_cheb = _nodCheb(np.array([0.0, 7.0]), 8)

    e5_poly_4_cheb = np.polyfit(e5_x_5_cheb, e5_f(e5_x_5_cheb), 4)
    e5_poly_6_cheb = np.polyfit(e5_x_7_cheb, e5_f(e5_x_7_cheb), 6)
    e5_poly_8_cheb = np.polyfit(e5_x_9_cheb, e5_f(e5_x_9_cheb), 8)

    e5_x_6 = np.linspace(0.0, 7.0, 6)

    e5_spline_3_natural = si.CubicSpline(e5_x_6, e5_f(e5_x_6), bc_type="natural")
    e5_spline_3_frontera = si.CubicSpline(e5_x_6, e5_f(e5_x_6), bc_type=((1, 0.209), (2, 0.0216)))

    e5_linspace = np.linspace(0.0, 7.0, 128)

    plt.plot(e5_linspace, e5_f(e5_linspace), color="red", label="Función")
    plt.plot(e5_linspace, np.polyval(e5_poly_4, e5_linspace), "-", color="blue", label="Poly4")
    plt.plot(e5_linspace, np.polyval(e5_poly_6, e5_linspace), "--", color="blue", label="Poly6")
    plt.plot(e5_linspace, np.polyval(e5_poly_8, e5_linspace), ".", color="blue", label="Poly8")
    plt.plot(e5_linspace, np.polyval(e5_poly_4_cheb, e5_linspace), "-", color="green", label="Poly4 (Cheb)")
    plt.plot(e5_linspace, np.polyval(e5_poly_6_cheb, e5_linspace), "--", color="green", label="Poly6 (Cheb)")
    plt.plot(e5_linspace, np.polyval(e5_poly_8_cheb, e5_linspace), ".", color="green", label="Poly8 (Cheb)")
    plt.plot(e5_linspace, e5_spline_3_natural(e5_linspace), color="purple", label="Spline Natural")
    plt.plot(e5_linspace, e5_spline_3_frontera(e5_linspace), "--", color="purple", label="Spline Frontera")
    plt.legend()
    plt.show()