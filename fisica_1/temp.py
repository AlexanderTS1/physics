import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

from scipy.integrate import odeint

log = logging.getLogger(__name__)

G = 9.8

def skate(y, t, params):
    phi, omega = y
    g, R = params
    derivs = [omega, -(g/R)*np.sin(phi)]
    return derivs

def small_angle_approximation_theta (theta0, w0, t):
    return theta0 * np.cos(w0 * t)

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
    
    log.info("Simulation started...")
    
    log.info("L = {0} [m]".format(args.l))
    log.info("M = {0} [kg]".format(args.m))
    log.info("Theta_0 = {0} [deg]".format(args.theta_0))
    log.info("t_0 = {0} [s]".format(args.t0))
    log.info("t_f = {0} [s]".format(args.tf))
    log.info("dt = {0} [s]".format(args.dt))
    
    
    ## SMALL-ANGLE APPROXIMATION
    t_values_ = []
    theta_values_approx_ = []
    
    t_ = args.t0
    w_0_ = np.sqrt(G / args.l)
    theta_0_ = np.radians(args.theta_0)
    
    while (t_ < args.tf):
        
        t_values_.append(t_)
        
        theta_t_ = small_angle_approximation_theta(theta_0_, w_0_, t_)
        theta_values_approx_.append(theta_t_)
        
        t_ += args.dt
        
    ## ODE INTEGRATION    
    w_0_ = 0.0
    theta_0_ = np.radians(args.theta_0)
    #t_values_ = np.linspace(args.t0, args.tf, int((args.tf - args.t0) / args.dt))
    z_0_ = [theta_0_, w_0_]
    
    z_ = odeint(skate, z_0_, t_values_, args=([G, args.l], ))
    
    ## Plotting
    
    plot(t_values_, theta_values_approx_, z_[:, 0])
    plot_difference(t_values_, theta_values_approx_, z_[:, 0])

if __name__ == "__main__":
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    parser_ = argparse.ArgumentParser(description="Parámetros")
    parser_.add_argument("--l", nargs="?", type=float, default=5.0, help="Longitud del péndulo [m]")
    parser_.add_argument("--m", nargs="?", type=float, default=1.0, help="Masa del péndulo [kg]")
    parser_.add_argument("--theta_0", nargs="?", type=float, default=20.0, help="Ángulo inicial [deg]")
    parser_.add_argument("--t0", nargs="?", type=float, default=0.0, help="Tiempo inicial [s]")
    parser_.add_argument("--tf", nargs="?", type=float, default=32.0, help="Tiempo final [s]")
    parser_.add_argument("--dt", nargs="?", type=float, default=0.01, help="Diferencial de tiempo [s]")
    
    args_ = parser_.parse_args()
    
    simulate(args_)