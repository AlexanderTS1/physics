import numpy as np
import matplotlib.pyplot as plt

from practicas.interpolacion import nodos_chebyshev
from practicas.aproximacion import w_chebyshev
from practicas.aproximacion import polinomios_ortogonales_chebyshev
from practicas.aproximacion import coeficientes_chebyshev
from practicas.aproximacion import aproximacion_polinomial
from practicas.aproximacion import polinomios_monicos

if __name__ == "__main__":

    # Ejercicio 1 -----------------------------------------------------------
    print("Ejercicio 1 ****************************************************")

    e1_f = lambda x: np.cos(np.arctan(x)) - np.exp(x**2.0) * np.log(x + 2.0)

    # a
    e1_nodos_chebyshev = nodos_chebyshev(np.array([-1.0, 1.0]), 5)
    e1_interpolador_cheb = np.polyfit(e1_nodos_chebyshev, e1_f(e1_nodos_chebyshev), 4)

    # b
    e1_pol_cheb = polinomios_monicos(polinomios_ortogonales_chebyshev(4))
    e1_a_cheb = coeficientes_chebyshev(e1_f, e1_pol_cheb)

    # c
    e1_linspace = np.linspace(-1.0, 1.0, 128)

    plt.plot(e1_linspace, e1_f(e1_linspace), color="red", label="Función")
    plt.plot(e1_linspace, np.polyval(e1_interpolador_cheb, e1_linspace), color="blue", label="Interpolador")
    plt.plot(e1_linspace, aproximacion_polinomial(e1_a_cheb, e1_pol_cheb, e1_linspace), color="green", label="Aproximante")
    plt.show()