# coding=utf-8
import argparse
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
#import matplotlib.animation as animation


def simular(args):
    fig=plt.figure()
    fig.set_dpi(100)

    matplotlib.rc('xtick', labelsize=40) 
    matplotlib.rc('ytick', labelsize=40) 
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size='40')
    #fig.set_size_inches(7,6.5)


    # Datos a modificar 
    # Circuito RLC en serie con un generador AC cuyo voltaje es V0*cos(omega*t)

    R=args.R  #resistencia
    C=args.C  #Capacidad del condensador
    L=args.L    #Inductancia de la bobina

    f=args.f   #frecuencia del generador
    Veficaz=args.V #voltaje eficaz del generador
    V0=Veficaz*np.sqrt(2.0) #voltaje máximo del generador
    omega=2.*math.pi*f  # frecuencia angular
    tf=args.tf  # tiempo de la simulacion

    par=[R,C,L]



    # Definicion de las ecuaciones de movimiento
    def circRCL(z,t,par):
        q,qp=z  
        dzdt=[qp,-q/C/L-R*qp/L+V0*math.cos(omega*t)/L]
        return dzdt


    # Llamada a odeint que resuelve las ecuaciones de movimiento

    nt=10000  #numero de intervalos de tiempo
    z0=[0.0,0.0] #Valores iniciales de Q e I    
    t=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    z=odeint(circRCL,z0,t,args=(par,),atol=abserr, rtol=relerr)


    plt.close('all')

    # Definicion del grafico
    # Realiza una gráfica del voltaje del generador y de la corriente del circuito
    # en función del tiempo. Cada curva tiene un eje y diferente

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Tiempo~$[s]$")

    ax1.set_ylabel(r"$I~[A]$", color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()

    ax1.set_xlim(xmin=0.08,xmax=.16) #limites del eje x


    line1, = ax1.plot(t[:],z[:,1],'--', linewidth=2, color='b')
    line2, = ax2.plot(t[:],V0*np.cos(omega*t[:]),'--', linewidth=2, color='r')

    ax2.set_ylabel(r"$V_{gen}~[V]$", color='r')
    ax2.tick_params('y', colors='r')

    fig.tight_layout()
    ax1.set_title(r"\textbf{Evolución de la Intensidad y del Voltaje}")
    plt.grid()

    plt.show()

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--C", nargs="?", type=float, default=75e-6, help="C")
    parser_.add_argument("--L", nargs="?", type=float, default=0.30, help="L")
    parser_.add_argument("--R", nargs="?", type=float, default=300, help="R")
    parser_.add_argument("--f", nargs="?", type=float, default=75, help="f")
    parser_.add_argument("--V", nargs="?", type=float, default=165, help="V")
    parser_.add_argument("--tf", nargs="?", type=float, default=2.0e-1, help="tf")

    args_ = parser_.parse_args()

    simular(args_)