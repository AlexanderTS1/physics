import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
#import matplotlib.animation as animation


#fig=plt.figure()
#fig.set_dpi(100)
#fig.set_size_inches(7,6.5)


####  Datos a modificar del programa

m=1.0e-3 / 6.022e23  #masa de la primer ion
q=1.6e-19  #carga del primer ion

m2=2.0e-3 / 6.022e23   #masa del segundo ion
q2=1.6e-19           #carga del segundo ion

m3=3.0e-3 / 6.022e23 #masa del tercer ion
q3=1.6e-19  #carga del tercer ion


Bf=1.0e-2 #Campo magnético usado
LE=10.0e-2  #Distancia entre las placas
dV=15.0  #Diferencia de potencial entre las placas

Ef=dV/LE  #Campo eléctrico
LB=0.25  #Distancia horizontal considerada

tf=18e-6  # tiempo de simulación considerado

##########################


qm=q/m
par=[qm,Ef,Bf,LE]

def espectrometro(z,t,par):
    x,vx,y,vy=z


    if y < LE :
      cB=0.
      cE=1.
    else:
      cB=1.
      cE=0.


    if x > 0.001 and y < LE :
        vx=0
        vy=0
    
    dzdt=[vx,qm*vy*cB*Bf,vy,qm*cE*Ef-qm*vx*cB*Bf]
    return dzdt


qm2=q2/m2
par2=[qm2,Ef,Bf,LE]


def espectrometro2(z2,t,par2):
    x2,vx2,y2,vy2=z2


    if y2 < LE :
      cB2=0.
      cE2=1.
    else:
      cB2=1.
      cE2=0.


    if x2 > 0.001 and y2 < LE :
        vx2=0
        vy2=0
    
    dz2dt=[vx2,qm2*vy2*cB2*Bf,vy2,qm2*cE2*Ef-qm2*vx2*cB2*Bf]
    return dz2dt


qm3=q3/m3
par3=[qm3,Ef,Bf,LE]


def espectrometro3(z3,t,par3):
    x3,vx3,y3,vy3=z3


    if y3 < LE :
      cB3=0.
      cE3=1.
    else:
      cB3=1.
      cE3=0.


    if x3 > 0.001 and y3 < LE :
        vx3=0
        vy3=0
    
    dz3dt=[vx3,qm3*vy3*cB3*Bf,vy3,qm3*cE3*Ef-qm3*vx3*cB3*Bf]
    return dz3dt


# Llamada a la subrutina odeint que resuelva las ecuaciones de movimiento
# para los tres iones definidos con los campos resultantes
# el tiempo varía entre 0 y tf

nt=10000
z0=[0.0,0.0,0.000000001,0.0]    
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z=odeint(espectrometro,z0,t,args=(par,),atol=abserr, rtol=relerr)
z2=odeint(espectrometro2,z0,t,args=(par2,),atol=abserr, rtol=relerr)
z3=odeint(espectrometro3,z0,t,args=(par3,),atol=abserr, rtol=relerr)

matplotlib.rc('xtick', labelsize=24) 
matplotlib.rc('ytick', labelsize=24) 
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size='32')

fig, ax = plt.subplots()


ax=plt.axes(xlim=(-0.2*LB,LB),ylim=(-0.2*LE,LB))


# Estas son las tres lineas de trayectorias que dibuja
# Cambiar los labels de acuerdo a lo calculado

line1, = ax.plot(z[:,0],z[:,2],'--', linewidth=2,
                 label=r'$^1H^+ (1 [g/mol])$ -- m_1 = ' + str(m) + r' [kg]', )
line2, = ax.plot(z2[:,0],z2[:,2], '--', linewidth=2,
                 label=r'$^2H^+ (1 [g/mol])$ -- m_2 = ' + str(m2) + r' [kg]')
line3, = ax.plot(z3[:,0],z3[:,2], '--', linewidth=2,
                 label=r'$^3H^+ (1 [g/mol])$ -- m_3 = ' + str(m3) + r' [kg]')


ax.legend(loc='upper center', fontsize=24)






# Create distintos rectangulos, flechas y circulo para el dibujo
rect = patches.Rectangle((-0.2*LB,0),0.4*LB,LE,linewidth=2,edgecolor='r',facecolor='none')
rect2 = patches.Rectangle((-0.02*LB,-0.05*LE),0.04*LB,LE*1.2,linewidth=2,edgecolor='none',facecolor='w')
rect3 = patches.Rectangle((-0.2*LB,LE*1.005),1.2*LB+0.1,0.6*LB,linewidth=2,edgecolor='g',facecolor='none')

arrow= patches.Arrow(0.1*LB,0.5*LE,0.0,0.2*LE,width=0.01)
arrow2= patches.Arrow(0.15*LB,0.5*LE,0.0,0.2*LE,width=0.01)
arrow3= patches.Arrow(-0.08*LB,0.5*LE,0.0,0.2*LE,width=0.01)
arrow4= patches.Arrow(-0.13*LB,0.5*LE,0.0,0.2*LE,width=0.01)

circle3=patches.Circle((-0.12*LB,LE+0.4*LB),0.05*LB,fill=False)
circle4=patches.Circle((-0.12*LB,LE+0.4*LB),0.01*LB)
circle5=patches.Circle((0.88*LB,LE+0.4*LB),0.05*LB,fill=False)
circle6=patches.Circle((0.88*LB,LE+0.4*LB),0.01*LB)
            


# agrega los Rectángulos, circulos, flechas y textos del dibujo
ax.add_patch(rect)
ax.add_patch(rect3)
ax.add_patch(rect2)
ax.add_patch(arrow)
ax.add_patch(arrow2)
ax.add_patch(arrow3)
ax.add_patch(arrow4)
ax.add_patch(circle3)
ax.add_patch(circle4)
ax.add_patch(circle5)
ax.add_patch(circle6)
ax.text(-0.05*LB,LE+0.38*LB, "B",fontsize=30)
ax.text(0.96*LB,LE+0.38*LB, "B",fontsize=30)
ax.text(0.1*LB, 0.3*LE, "E", fontsize=30)
ax.text(-0.12*LB, 0.3*LE, "E", fontsize=30)

#labels de los ejes x e y 
ax.set_ylabel("y [m]", fontsize=32)
ax.set_xlabel("x [m]", fontsize=32)


# Esta parte agrega un grafico pequeño para ver mejor la posicion donde llegan los iones
# These are in unitless percentages of the figure size. (0,0 is bottom left)
# CAMBIAR
left, bottom, width, height = [0.6, 0.2, 0.25, 0.2]
ax = fig.add_axes([left, bottom, width, height])

line4, = ax.plot(z[:,0],z[:,2], '--', linewidth=2)
line5, = ax.plot(z2[:,0],z2[:,2], '--', linewidth=2)
line6, = ax.plot(z3[:,0],z3[:,2], '--', linewidth=2)

#intervalos de los ejes x e y para la grafica pequeña.
#CAMBIAR
ax.set_xlim(0.10,0.2)
ax.set_ylim(0.10,0.12)

#Donde pone los ticks y titulos de los ejes de la grafica pequeña
plt.xticks([0.1,0.15,0.2])
plt.yticks([0.10,0.11,0.12])
ax.set_ylabel("y [m]", fontsize=32)
ax.set_xlabel("x [m]", fontsize=32)

plt.show()
