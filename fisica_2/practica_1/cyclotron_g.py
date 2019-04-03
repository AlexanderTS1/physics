import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from scipy import signal
#import matplotlib.animation as animation
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot

#fig=plt.figure()
#fig, axarr=plt.subplots(1,2)
#fig.set_dpi(100)
#fig.set_size_inches(7,6.5)


# Este programa calcula la trayectoria de un determinado ion en un ciclotron
# Y grafica como va cambiando la energia cinetica del ion con el tiempo

#Datos de entrada del programa

m=1.67e-27  # masa del ion
q=1.60e-19   #carga del ion
qm=q/m      # Relacion carga masa
Bf=1e-1    # Campo magnético existente en las Ds
dV=5e3   # Diferencia de potencial alterna existente entre las Ds
dE=2e-2      # Separacion entre las Ds 
# dE debe ser pequeño para que el tiempo que el ion pasa entre las Ds sea 
# despreciable frente al tiempo que el ion recorre en las Ds

Ef=dV/dE     #Campo electrico existente entre las Ds del ciclotron
omega=qm*Bf #frecuencia angular del voltaje entre las Ds 
# si omega=qm*Bf esta es la frec. del ciclotron y la señal estara en fase
# con el movimiento del ion


Rc=2.5e-1      #Radio de las Ds del ciclotron

tf=49e-7   # tiempo final de simulacion


par=[qm,Ef,Bf,dE,Rc,omega]


# Definicion de las ecuaciones de movimiento del ion
# El voltaje entre las Ds es una señal cuadrada con frecuencia
# angular omega y valor Ef, aunque existe la opcion de poner
# una señal sinusoidal
 

def ciclotron(z,t,par):
    x,vx,y,vy=z

    yy=math.fabs(y)

    if yy < 0.5*dE :
      cB=0.
      cE=1.
    else:
      cB=1.
      cE=0.

    r=math.sqrt((yy-0.5*dE)*(yy-0.5*dE)+x*x)
#    coswt=math.cos(omega*t)
#    s=math.copysign(1,vy)
    cuad=signal.square(omega*t)


    if r > Rc :
        cE=0.
        cB=0.
    
#    dzdt=[vx,qm*vy*cB*Bf,vy,qm*cE*Ef*coswt-qm*vx*cB*Bf]
#     dzdt=[vx,qm*vy*cB*Bf,vy,qm*cE*Ef*s-qm*vx*cB*Bf]
    dzdt=[vx,qm*vy*cB*Bf,vy,qm*cE*Ef*cuad-qm*vx*cB*Bf]
    return dzdt




# Llamada a odeint que resuelve ec. Movimiento del ion

nt=10000
z0=[0.0,0.0,0.0,0.0]    
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z=odeint(ciclotron,z0,t,args=(par,),atol=abserr, rtol=relerr)


matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size='20')

plt.close('all')


# Definicion de las graficas
# Modificar los limites de los ejes a gusto, así como los titulos

f0 = figure(num = 0, figsize = (5, 10))#, dpi = 100)
ax01 = subplot2grid((2, 2), (0, 0),colspan=2)
ax02 = subplot2grid((2, 2), (1, 0),colspan=2)
ax01.grid(True)
ax02.grid(True)

# Limites de la grafica del ciclotron
ax01.set_xlim(-1.5*Rc,1.5*Rc)
ax01.set_ylim(-1.2*Rc,1.2*Rc)

# Limites de la grafica de la Ec(t)
# Modificarlo para ver la grafica
ax02.set_xlim(0,tf)

ax02.set_xlabel("Tiempo [s]")
ax02.set_ylabel("Energía Cinética [J]")

# La primera linea corresponde a la trayectoria del ion en el ciclotron
# la segunda linea corresponde a la grafica de la energía cinetica en funcion del tiempo
# Modificar esta segunda linea si se quiere graficar otra variable o se pone la Ec en 
# otras unidades 

line, = ax01.plot(z[:,0],z[:,2], linewidth=2)
line2, = ax02.plot(t[:],m*(z[:,1]**2+z[:,3]**2)/2, linewidth=2)


ax02.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
ax02.ticklabel_format(axis='x', style='sci', scilimits=(-2,2))




# Crea los diferentes circulos, arcos, lineas y textos del dibujo

circle1=patches.Circle((-0.7*Rc,0.3*Rc),0.05*Rc,fill=False)
circle2=patches.Circle((-0.7*Rc,0.3*Rc),0.01*Rc)
circle3=patches.Circle((-0.7*Rc,-0.3*Rc),0.05*Rc,fill=False)
circle4=patches.Circle((-0.7*Rc,-0.3*Rc),0.01*Rc)


pac = patches.Arc([0,0.5*dE], 2*Rc, 2*Rc, angle=0, theta1=0, theta2=180)
pac2 = patches.Arc([0,-0.5*dE], 2*Rc, 2*Rc, angle=180, theta1=0, theta2=180)
line1=patches.Arrow(-Rc,0.5*dE,2.15*Rc,0.,width=0)
line2=patches.Arrow(-Rc,-0.5*dE,2.15*Rc,0.,width=0)
circle5=patches.Circle((1.28*Rc,0),0.14*Rc,fill=False)
line3=patches.Arrow(1.17*Rc,0.03*Rc,0.05*Rc,0,width=0)
line4=patches.Arrow(1.22*Rc,0.03*Rc,0,-0.08*Rc,width=0)
line5=patches.Arrow(1.22*Rc,-0.05*Rc,0.05*Rc,0,width=0)
line6=patches.Arrow(1.27*Rc,-0.05*Rc,0,0.08*Rc,width=0)
line7=patches.Arrow(1.27*Rc,0.03*Rc,0.05*Rc,0,width=0)
line8=patches.Arrow(1.32*Rc,0.03*Rc,0,-0.08*Rc,width=0)
line9=patches.Arrow(1.32*Rc,-0.05*Rc,0.05*Rc,0,width=0)
line10=patches.Arrow(1.37*Rc,-0.05*Rc,0,0.08*Rc,width=0)

            

# Add the patch to the Axes
ax01.add_patch(circle1)
ax01.add_patch(circle2)
ax01.add_patch(circle3)
ax01.add_patch(circle4)
ax01.add_patch(circle5)

ax01.text(-0.65*Rc,0.3*Rc, "B",fontsize=32)
ax01.text(-0.65*Rc,-0.3*Rc, "B",fontsize=32)


ax01.add_patch(pac)
ax01.add_patch(pac2)
ax01.add_patch(line1)
ax01.add_patch(line2)
ax01.add_patch(line3)
ax01.add_patch(line4)
ax01.add_patch(line5)
ax01.add_patch(line6)
ax01.add_patch(line7)
ax01.add_patch(line8)
ax01.add_patch(line9)
ax01.add_patch(line10)

#plt.figure()   #Anade un nuevo grfico y lo activa
#plot(t[0:nt],z[0:nt,0], "r")
ax01.set_xlabel("x [m]")
ax01.set_ylabel("y [m]")
#ax02.set_title("Grafico")


plt.show()



