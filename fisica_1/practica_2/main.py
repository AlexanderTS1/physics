import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.integrate import odeint

log = logging.getLogger(__name__)

G = 9.8

def pendulum(y, t, params):
    phi, omega = y
    g, R, alpha = params
    derivs = [omega, -(g/R)*np.sin(phi) - alpha * omega]
    return derivs

def small_angle_approximation_theta (theta0, w0, t):
    return theta0 * np.cos(w0 * t)

def time_to_diff(tValues, thetaNumeric, thetaAnalytic, threshold=0.01):

    for i in range(len(thetaNumeric)):
        if abs((thetaNumeric[i] - thetaAnalytic[i])/thetaAnalytic[i]) > 0.01:
            return tValues[i]

    return None

def plot_list(tValues, values, yTicks, yLabel, legends, loc='right'):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='42')
    
    plt.subplot(1, 1, 1)

    for i in range(len(values)):
        plt.plot(tValues[i], values[i], "-", linewidth=2)

    plt.xlabel(r"Tiempo $[s]$")
    plt.ylabel(yLabel)
    plt.yticks(yTicks)
    plt.legend(legends, loc=loc)
    plt.grid()

    plt.show()

def plot(tValues, thetaValuesApprox, thetaValuesIntegration, legend, loc='lower left'):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='42')

    theta_values_approx_deg_ = np.degrees(np.array(thetaValuesApprox))
    theta_values_integration_deg_ = np.degrees(np.array(thetaValuesIntegration))
    
    plt.subplot(1, 1, 1)
    plt.plot(tValues, theta_values_approx_deg_, "-", linewidth=2)
    plt.plot(tValues, theta_values_integration_deg_, "-", linewidth=2)
    plt.xlabel(r"Tiempo $[s]$")
    plt.ylabel(r"$\theta [deg]$")
    plt.legend([legend + " Numérico", legend + " Analítico"], loc=loc)
    plt.grid()

    plt.show()
    
def plot_difference(tValues, thetaValuesApprox, thetaValuesIntegration, legend):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='42')

    theta_values_approx_deg_ = np.degrees(np.array(thetaValuesApprox))
    theta_values_integration_deg_ = np.degrees(np.array(thetaValuesIntegration))
    
    theta_values_diff_ = theta_values_integration_deg_ - theta_values_approx_deg_
    
    plt.subplot(1, 1, 1)
    plt.plot(tValues, theta_values_diff_, "-", linewidth=2)
    plt.xlabel(r"Tiempo $[s]$")
    plt.ylabel(r"$\theta [deg]$")
    plt.grid()
    
    plt.show()

def plot_difference_list(tValues, thetaValuesApprox, thetaValuesIntegration, legends, loc='upper right'):
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='42')

    plt.subplot(1, 1, 1)

    for i in range(len(thetaValuesApprox)):
        theta_values_approx_deg_ = np.degrees(np.array(thetaValuesApprox[i]))
        theta_values_integration_deg_ = np.degrees(np.array(thetaValuesIntegration[i]))
        
        theta_values_diff_ = theta_values_integration_deg_ - theta_values_approx_deg_
        
        plt.plot(tValues, theta_values_diff_, "-", linewidth=2)
        
    plt.xlabel(r"Tiempo $[s]$")
    plt.ylabel(r"$\theta [deg]$")
    plt.legend(legends, loc=loc)
    plt.grid()
    
    plt.show()

def simulate(args):

    t_values_list_ = []
    theta_values_approx_list_ = []
    theta_values_list_ = []
    omega_values_list_ = []
    tension_values_list_ = []
    tension_values_approx_list_ = []
    legend_list_ = []
    
    for theta_0 in args.theta_0:

        for alpha in args.alpha:

            log.info("Simulation started...")
            
            log.info("L = {0} [m]".format(args.l))
            log.info("M = {0} [kg]".format(args.m))
            log.info("Theta_0 = {0} [deg]".format(theta_0))
            log.info("alpha = {0}".format(alpha))
            log.info("t_0 = {0} [s]".format(args.t0))
            log.info("t_f = {0} [s]".format(args.tf))
            log.info("dt = {0} [s]".format(args.dt))
            
            ## SMALL-ANGLE APPROXIMATION
            t_values_ = []
            theta_values_approx_ = []
            
            t_ = args.t0
            w_0_ = np.sqrt(G / args.l)
            theta_0_ = np.radians(theta_0)
            
            while (t_ < args.tf):
                
                t_values_.append(t_)
                
                theta_t_ = small_angle_approximation_theta(theta_0_, w_0_, t_)
                theta_values_approx_.append(theta_t_)
                
                t_ += args.dt

            tension_approx_ = args.m * G * np.cos(theta_values_approx_)
                
            ## ODE INTEGRATION    
            w_0_ = 0.0
            theta_0_ = np.radians(theta_0)
            #t_values_ = np.linspace(args.t0, args.tf, int((args.tf - args.t0) / args.dt))
            z_0_ = [theta_0_, w_0_]
            z_ = odeint(pendulum, z_0_, t_values_, args=([G, args.l, alpha], ))
            tension_ = args.m * G * np.cos(z_[:, 0])

            # Add values to list
            t_values_list_.append(t_values_)
            theta_values_approx_list_.append(theta_values_approx_)
            theta_values_list_.append(z_[:, 0])
            omega_values_list_.append(z_[:, 1])
            tension_values_list_.append(tension_)
            tension_values_approx_list_.append(tension_approx_)
            #legend_list_.append(r"$\theta_0$=" + str(theta_0))
            #legend_list_.append(r"$\theta_0$=" + str(theta_0) + r" $\alpha$=" + str(alpha))
            legend_list_.append(r" $\alpha$=" + str(alpha))

            # Time to 1% difference
            ttd_ = time_to_diff(t_values_, z_[:, 0], theta_values_approx_, 0.01)
            ttd2_ = time_to_diff(t_values_, theta_values_approx_, z_[:, 0], 0.01)
            log.info("Time until solutions differ 1%: {0}".format(ttd_))
            log.info("Time until solutions differ 2 1%: {0}".format(ttd2_))

            # Plotting
            plot(t_values_, theta_values_approx_, z_[:, 0], r"$\theta_0$=" + str(theta_0))
            #plot_difference(t_values_, theta_values_approx_, z_[:, 0])
        
        ## Plotting
        plot_list(t_values_list_, np.degrees(np.array(theta_values_list_)), np.arange(-50, 60, step=10), r"$\theta [deg]$", legend_list_)
        plot_list(t_values_list_, omega_values_list_, np.arange(-1, 1.5, step=0.5), r"$\omega [rad/s]$", legend_list_)
        plot_list(t_values_list_, tension_values_list_, np.arange(0, 11, step=1), r"$T [N]$", legend_list_, loc='lower center')
        plot_list(t_values_list_, tension_values_approx_list_, np.arange(0, 11, step=1), r"$T [N]$", legend_list_, loc='lower center')
        plot_difference_list(t_values_, theta_values_approx_list_, theta_values_list_, legend_list_)


if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    parser_ = argparse.ArgumentParser(description="Parámetros")
    parser_.add_argument("--l", nargs="?", type=float, default=5.0, help="Longitud del péndulo [m]")
    parser_.add_argument("--m", nargs="?", type=float, default=1.0, help="Masa del péndulo [kg]")
    parser_.add_argument("--theta_0", nargs="+", type=float, default=[5.0], help="Ángulo inicial [deg]")
    parser_.add_argument("--alpha", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.8], help="Constante de rozamiento")
    parser_.add_argument("--t0", nargs="?", type=float, default=0.0, help="Tiempo inicial [s]")
    parser_.add_argument("--tf", nargs="?", type=float, default=10.0, help="Tiempo final [s]")
    parser_.add_argument("--dt", nargs="?", type=float, default=0.01, help="Diferencial de tiempo [s]")
    
    args_ = parser_.parse_args()
    
    simulate(args_)