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
    # Este programa estudia a un circuito RLC sin generador. Se parte de un condensador
    # con carga Q0 y se estudia como varía la carga en función del tiempo o la corriente
    
    # R=1.0  # Resistencia del circuito
    # C=1.E-6 # Capacidad del condensador
    # Q0=1.E-6 #Carga inicial que tiene el condensador
    # L=1.E-3 # Inductancia de la bobina
    # tf=1.E-3 # tiempo final de simulación

    # # Oscilador no amortiguado
    # R = 0.0
    # C = 75.0e-6
    # L = 0.15
    # Q0 = 5.0e-6
    # tf = 1.0e-1

    # # Oscilador amortiguado
    # C = 75.0e-6
    # L = 0.15
    # R = np.sqrt(4.0 * L / C) * 1e-1
    # Q0 = 5.0e-6
    # tf = 1.0e-1

    # # Oscilador sobreamortiguado
    # C = 75.0e-6
    # L = 0.15
    # R = np.sqrt(4.0 * L / C) * 1e1
    # Q0 = 5.0e-6
    # tf = 1.0e-1
    L = args.L
    C = args.C
    R = np.sqrt(4.0 * L / C) * args.R
    Q0 = args.Q0
    tf = args.tf

    par=[R,C,L]

    # Definicion de las ecuaciones dinámicas del sistema
    def circRCL(z,t,par):
        q,qp=z  
        dzdt=[qp,-q/C/L-R*qp/L]
        return dzdt


    # Llamada a odeint que resuleve las ecuaciones de movimiento

    nt=10000
    z0=[Q0,0.0]    
    t=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    z=odeint(circRCL,z0,t,args=(par,),atol=abserr, rtol=relerr)

    ue_ = 0.5 * np.square(z[:, 0]) / C
    um_ = 0.5 * L * np.square(z[:, 1])
    u_ = ue_ + um_

    ur_ = np.zeros_like(u_)
    for i in range(len(t)-1):
        ur_[i+1] = ur_[i] + z[i+1,1] * z[i+1,1] * R

    # Definicion del grafico
    # Modificar los limites de acuerdo a las necesidades, así como los titulos de los ejes
    # está puesto para que grafique Q=f(t)  (z[0]->Q   z[1]->I)

    ax=plt.axes(xlim=(0,tf))
    line1, = ax.plot(t[:],z[:,0],'-', linewidth=4)
    #ax.legend(loc='lower right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.set_ylabel(r"\textbf{$Q [C]$}")
    ax.set_xlabel(r"\textbf{$t [s]$}")
    ax.set_title(r"\textbf{Evolución de la Carga}")
    plt.grid()
    plt.show()

    ax=plt.axes(xlim=(0,tf))
    line1, = ax.plot(t[:],z[:,1],'-', linewidth=4)
    #ax.legend(loc='lower right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.set_ylabel(r"\textbf{$I [A]$}")
    ax.set_xlabel(r"\textbf{$t [s]$}")
    ax.set_title(r"\textbf{Evolución de la Intensidad}")
    plt.grid()
    plt.show()

    ax=plt.axes(xlim=(0,tf))
    line1, = ax.plot(t[:],u_,'-', linewidth=4, label=r"$U$")
    line2, = ax.plot(t[:],ue_,'-', linewidth=4, label=r"$U_e$")
    line3, = ax.plot(t[:],um_,'-', linewidth=4, label=r"$U_m$")
    ax.legend(loc='lower right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.set_ylabel(r"\textbf{$U [J]$}")
    ax.set_xlabel(r"\textbf{$t [s]$}")
    ax.set_title(r"\textbf{Evolución de la Energía}")
    plt.grid()
    plt.show()

    ax=plt.axes(xlim=(0,tf))
    line1, = ax.plot(t[:],ur_,'-', linewidth=4, label=r"$U_R$")
    ax.legend(loc='lower right')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.set_ylabel(r"\textbf{$U_R [J]$}")
    ax.set_xlabel(r"\textbf{$t [s]$}")
    ax.set_title(r"\textbf{Evolución de la Energía Disipada}")
    plt.grid()
    plt.show()

if __name__ == "__main__":

    parser_ = argparse.ArgumentParser(description="Parameters")
    parser_.add_argument("--C", nargs="?", type=float, default=75.0e-6, help="C")
    parser_.add_argument("--L", nargs="?", type=float, default=0.15, help="L")
    parser_.add_argument("--Q0", nargs="?", type=float, default=5.0e-6, help="Q0")
    parser_.add_argument("--R", nargs="?", type=float, default=0, help="R")
    parser_.add_argument("--tf", nargs="?", type=float, default=1.0e-1, help="tf")

    args_ = parser_.parse_args()

    simular(args_)