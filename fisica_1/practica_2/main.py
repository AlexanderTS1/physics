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

def plot(tValues, thetaValuesApprox, thetaValuesIntegration):
    
    theta_values_approx_deg_ = np.degrees(np.array(thetaValuesApprox))
    theta_values_integration_deg_ = np.degrees(np.array(thetaValuesIntegration))
    
    plt.subplot(1, 1, 1)
    plt.plot(tValues, theta_values_approx_deg_, "-")
    plt.plot(tValues, theta_values_integration_deg_, "-")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [deg]")
    
    plt.show()
    
def plot_difference(tValues, thetaValuesApprox, thetaValuesIntegration):
    
    theta_values_approx_deg_ = np.degrees(np.array(thetaValuesApprox))
    theta_values_integration_deg_ = np.degrees(np.array(thetaValuesIntegration))
    
    theta_values_diff_ = theta_values_integration_deg_ - theta_values_approx_deg_
    
    plt.subplot(1, 1, 1)
    plt.plot(tValues, theta_values_diff_, "-")
    plt.xlabel("Time [s]")
    plt.ylabel("Theta [deg]")
    
    plt.show()

def simulate(args):

    t_values_list_ = []
    theta_values_approx_list_ = []
    theta_values_list_ = []
    omega_values_list_ = []
    tension_values_list_ = []
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
                
            ## ODE INTEGRATION    
            w_0_ = 0.0
            theta_0_ = np.radians(theta_0)
            #t_values_ = np.linspace(args.t0, args.tf, int((args.tf - args.t0) / args.dt))
            z_0_ = [theta_0_, w_0_]
            z_ = odeint(pendulum, z_0_, t_values_, args=([G, args.l, alpha], ))
            tension_ = args.m * G * np.cos(z_[:, 0])

            # Plotting
            #plot(t_values_, theta_values_approx_, z_[:, 0])
            #plot_difference(t_values_, theta_values_approx_, z_[:, 0])

            # Add values to list
            t_values_list_.append(t_values_)
            theta_values_approx_list_.append(theta_values_approx_)
            theta_values_list_.append(z_[:, 0])
            omega_values_list_.append(z_[:, 1])
            tension_values_list_.append(tension_)
            legend_list_.append(r"$\theta_0$=" + str(theta_0))
            #legend_list_.append(r"$\theta_0$=" + str(theta_0) + r" $\alpha$=" + str(alpha))
            #legend_list_.append(r" $\alpha$=" + str(alpha))
        
        ## Plotting
        plot_list(t_values_list_, np.degrees(np.array(theta_values_list_)), np.arange(-50, 60, step=10), r"$\theta [deg]$", legend_list_)
        plot_list(t_values_list_, omega_values_list_, np.arange(-1, 1.5, step=0.5), r"$\omega [rad/s]$", legend_list_)
        plot_list(t_values_list_, tension_values_list_, np.arange(0, 11, step=1), r"$T [N]$", legend_list_, loc='bottom center')

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