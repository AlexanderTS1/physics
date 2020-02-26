import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate


if __name__ == "__main__":

  np.random.seed(0)

  # Ejercicio 1 -------------------------------------------------------------
  print("Ejercicio 1 ******************************************************")

  p = np.array([1.0, -1.0, 3.0, 2.0, 5.0])

  p_polyval_1 = np.polyval(p, 1)
  p_polyval_0 = np.polyval(p, 0)

  print(p_polyval_1)
  print(p_polyval_0)

  e1_poly = np.array([1.0, 0.0, -0.25, 1.0, 2.0])

  print("Valor del polinomio en x=1 es {}".format(np.polyval(e1_poly, 1.0)))
  print("Valor del polinomio en x=-2 es {}".format(np.polyval(e1_poly, -2.0)))

  # Ejercicio 2 -------------------------------------------------------------
  print("Ejercicio 2 ******************************************************")

  e2_degree = 4
  e2_x_values = np.linspace(-1.0, 1.0, e2_degree + 1)
  e2_f = lambda x: 1.0 / (1.0 + x**2)
  e2_y_values = e2_f(e2_x_values)
  e2_fit = np.polyfit(e2_x_values, e2_y_values, e2_degree)

  e2_linspace = np.linspace(-1.0, 1.0, 128)

  #plt.scatter(e2_x_values, e2_y_values, color="red", marker='x')
  #plt.plot(e2_x_values, e2_y_values, color="blue", marker='o')
  #plt.plot(e2_linspace, np.polyval(e2_fit, e2_linspace), color="green")
  #plt.xlabel("Abscisas")
  #plt.ylabel("Ordenadas")
  #plt.legend(["Puntos", "Función", "Polinomio"])
  #plt.show()

  print(e2_fit)

  # Ejercicio 3 -------------------------------------------------------------
  print("Ejercicio 3 ******************************************************")

  def _pol_lagrange(
      x: np.array,
      y: np.array
  ) -> np.array:

    vandermonde = np.vander(x)
    return np.matmul(np.linalg.inv(vandermonde), y)

  # Ejercicio 4 -------------------------------------------------------------
  print("Ejercicio 4 ******************************************************")

  e4_num_points = 6
  e4_x_values = np.sort(np.random.uniform(-5.0, 4.0, e4_num_points))
  e4_y_values = np.sort(np.random.uniform(-2.0, 6.0, e4_num_points))

  print(e4_x_values)
  print(e4_y_values)

  e4_pol_lagrange = _pol_lagrange(e4_x_values, e4_y_values)
  e4_fit = np.polyfit(e4_x_values, e4_y_values, e4_num_points-1)

  e4_linspace = np.linspace(-5.0, 4.0, 128)
  # plt.scatter(e4_x_values, e4_y_values, color="red", marker='x')
  # plt.plot(e4_linspace, np.polyval(e4_pol_lagrange, e4_linspace), color="green")
  # plt.plot(e4_linspace, np.polyval(e4_fit, e4_linspace), "--", color="blue")
  # plt.xlabel("Abscisas")
  # plt.ylabel("Ordenadas")
  # plt.legend(["Polinomio Lagrange", "Polyfit", "Puntos"])
  # plt.ylim([-2.0, 6.0])
  # plt.show()

  # Ejercicio 5 -------------------------------------------------------------
  print("Ejercicio 5 ******************************************************")

  e5_f = lambda x: np.exp(x) * np.cos(3.0 * x)
  e5_x_values = np.array([-1.5, -0.75, 0.0, 1.0, 1.5, 2.0, 2.7])
  e5_y_values = e5_f(e5_x_values)

  print(e5_y_values)

  e5_pol_lagrange = _pol_lagrange(e5_x_values, e5_y_values)

  e5_linspace = np.linspace(-2.0, 3.0, 128)
  # plt.scatter(e5_x_values, e5_y_values, color="red", marker='x')
  # plt.plot(e5_linspace, e5_f(e5_linspace), color="green")
  # plt.plot(e5_linspace, np.polyval(e5_pol_lagrange, e5_linspace), "--", color="blue")
  # plt.xlabel("Abscisas")
  # plt.ylabel("Ordenadas")
  # plt.legend(["Polinomio Lagrange", "Función", "Puntos"])
  # plt.show()

  # Ejercicio 6 -------------------------------------------------------------
  print("Ejercicio 6 ******************************************************")

  e6_degrees = [6, 8, 10]
  e6_f = lambda x: np.cos(x)**5

  for degree in e6_degrees:

    e6_x_values = np.linspace(0.0, 2.0, degree + 1)
    e6_y_values = e6_f(e6_x_values)

    e6_fit = np.polyfit(e6_x_values, e6_y_values, degree)

    e6_error = np.abs(
        np.polyval(
            e6_fit, e6_x_values
        ) - e6_y_values
    )

    e6_linspace = np.linspace(0.0, 2.0, 128)

    # plt.scatter(e6_x_values, e6_y_values, color="red", marker='x')
    # plt.plot(e6_linspace, e6_f(e6_linspace), color="green")
    # plt.plot(e6_linspace, np.polyval(e6_fit, e6_linspace), "--", color="blue")
    # plt.xlabel("Abscisas")
    # plt.ylabel("Ordenadas")
    # plt.legend(["Polyfit", "Función", "Puntos"])
    # plt.xlim([0.0, 2.0])
    # plt.show()

    # plt.plot(e6_x_values, e6_error)
    # plt.show()


  # Ejercicio 7 -------------------------------------------------------------
  print("Ejercicio 7 ******************************************************")

  e7_x_values = 0.2 * np.linspace(0.0, 5.0, 5)
  e7_f = lambda x: (2.0 / np.sqrt(np.pi)) * integrate.quad(lambda y: np.exp(-y**2), 0.0, x)[0]
  
  e7_y_values = np.zeros(len(e7_x_values))
  for i in range(len(e7_x_values)):
    e7_y_values[i] = e7_f(e7_x_values[i])

  e7_fit_1 = np.polyfit(e7_x_values, e7_y_values, 1)
  e7_fit_2 = np.polyfit(e7_x_values, e7_y_values, 2)

  e7_linspace = 0.2 * np.linspace(0.0, 5.0, 128)

  # e7_f_values = []
  # for x in e7_linspace:
  # 	e7_integral = integrate.quad(lambda y: np.exp((-x)**2), 0.0, x)
  # 	e7_f_values.append(2.0 / np.sqrt(np.pi) * e7_integral[0])
  # e7_f_values = np.array(e7_f_values)

  e7_f_values = np.zeros(len(e7_linspace))
  for i in range(len(e7_linspace)):
    e7_f_values[i] = e7_f(e7_linspace[i])

  # plt.scatter(e7_x_values, e7_y_values, color="red", marker='x')
  # plt.plot(e7_linspace, e7_f_values, color="green")
  # plt.plot(e7_linspace, np.polyval(e7_fit_1, e7_linspace), "--", color="blue")
  # plt.plot(e7_linspace, np.polyval(e7_fit_2, e7_linspace), "--", color="yellow")
  # plt.xlabel("Abscisas")
  # plt.ylabel("Ordenadas")
  # plt.legend(["Cuadrático", "Lineal", "Función", "Puntos"])
  # plt.show()

  e7_13_fit_1 = np.polyfit(e7_x_values[1:3], e7_y_values[1:3], 1)
  e7_13_fit_2 = np.polyfit(e7_x_values[0:3], e7_y_values[0:3], 2)

  print("Error lineal {}".format(
      np.abs(np.polyval(e7_13_fit_1, 1.0/3.0) - e7_f(1.0/3.0)))
  )

  print("Error cuadratico {}".format(
      np.abs(np.polyval(e7_13_fit_2, 1.0/3.0) - e7_f(1.0/3.0)))
  )

  # Ejercicio 8 -------------------------------------------------------------
  print("Ejercicio 8 ******************************************************")

  def _dif_divididas(
      x: np.array,
      y: np.array
  ) -> np.array:


    dif_div = np.zeros([len(x), len(x)])
    dif_div[:, 0] = y

    for j in range(1, len(x)):

      for i in range(len(x) - j):

        dif_div[i, j] = (dif_div[i+1, j-1] - dif_div[i, j-1]) / (x[j+i] - x[i])

    return dif_div[0]
      

  # Ejercicio 9 -------------------------------------------------------------
  print("Ejercicio 9 ******************************************************")

  def _pol_newton(
      x: np.array,
      y: np.array,
      grado: int
  ):

    dif_div = _dif_divididas(x, y)

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


  # Ejercicio 10 ------------------------------------------------------------
  print("Ejercicio 10 *****************************************************")

  e10_x_values = np.array([1.0, 2.0, 4.0, 6.0])
  e10_y_values = np.array([2.0, 4.0, 6.0, 5.0])

  e10_newton = _pol_newton(e10_x_values, e10_y_values, 3)
  e10_polyfit = np.polyfit(e10_x_values, e10_y_values, 2)
  e10_lagrange = _pol_lagrange(e10_x_values, e10_y_values)

  e10_linspace = np.linspace(1.0, 6.0, 128)

  e10_newton_values = np.zeros(len(e10_linspace))
  for i in range(len(e10_linspace)):
    e10_newton_values[i] = e10_newton(e10_linspace[i])

  # plt.scatter(e10_x_values, e10_y_values, color="red", marker='x', label="Puntos")
  # plt.plot(e10_linspace, e10_newton_values, color="green", label="Newton")
  # plt.plot(e10_linspace, np.polyval(e10_polyfit, e10_linspace), "--", color="blue", label="Cuadrático")
  # plt.plot(e10_linspace, np.polyval(e10_lagrange, e10_linspace), "--", color="yellow", label="Lagrange")
  # plt.xlabel("Abscisas")
  # plt.ylabel("Ordenadas")
  # plt.legend()
  # plt.show()

  print("Ejercicio 11 *****************************************************")

  e11_x_values = np.array([-1.5, -0.75, 0.0, 1.0, 1.5, 2.0, 2.7])
  e11_f = lambda x: np.exp(x) * np.cos(3.0 * x)
  e11_y_values = e11_f(e11_x_values)

  e11_newton = _pol_newton(e11_x_values, e11_y_values, 6)
  e11_polyfit = np.polyfit(e11_x_values, e11_y_values, 6)
  e11_lagrange = _pol_lagrange(e11_x_values, e11_y_values)

  e11_linspace = np.linspace(-1.5, 2.7, 128)

  e11_newton_values = np.zeros(len(e11_linspace))
  for i in range(len(e11_linspace)):
    e11_newton_values[i] = e11_newton(e11_linspace[i])

  plt.scatter(e11_x_values, e11_y_values, color="red", marker='x', label="Puntos")
  plt.plot(e11_linspace, e11_newton_values, color="green", label="Newton")
  plt.plot(e11_linspace, np.polyval(e11_polyfit, e11_linspace), "--", color="blue", label="Cuadrático")
  plt.plot(e11_linspace, np.polyval(e11_lagrange, e11_linspace), "--", color="yellow", label="Lagrange")
  plt.xlabel("Abscisas")
  plt.ylabel("Ordenadas")
  plt.legend()
  plt.show()