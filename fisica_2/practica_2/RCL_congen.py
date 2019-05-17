# coding=utf-8
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
#import matplotlib.animation as animation


fig=plt.figure()
fig.set_dpi(100)
#fig.set_size_inches(7,6.5)


# Datos a modificar 
# Circuito RLC en serie con un generador AC cuyo voltaje es V0*cos(omega*t)

R=100.0  #resistencia
C=1.E-5  #Capacidad del condensador
L=0.1    #Inductancia de la bobina

f=50.0   #frecuencia del generador
Veficaz=220.0 #voltaje eficaz del generador
V0=Veficaz*np.sqrt(2.0) #voltaje máximo del generador
omega=2.*math.pi*f  # frecuencia angular
tf=5.E-1  # tiempo de la simulacion

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

ax1.set_xlabel('time (s)')

ax1.set_ylabel('I (A)', color='b')
ax1.tick_params('y', colors='b')

ax2 = ax1.twinx()

ax1.set_xlim(xmin=0.08,xmax=.16) #limites del eje x


line1, = ax1.plot(t[:],z[:,1],'--', linewidth=2, color='b')
line2, = ax2.plot(t[:],V0*np.cos(omega*t[:]),'--', linewidth=2, color='r')


ax2.set_ylabel('Vgen (V)', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()

plt.show()





