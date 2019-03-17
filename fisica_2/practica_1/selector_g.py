import numpy as np
import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.mlab as mlab
from scipy.integrate import odeint
from matplotlib.font_manager import FontProperties
#import matplotlib.animation as animation


#fig=plt.figure()
#fig.set_dpi(100)
#fig.set_size_inches(7,6.5)

# Datos de entrada
# El programa calcula la trayectoria en un selector de velocidad de un ion con una
# dada velocidad, y la del mismo ion pero con una velocidad 10 % mayor y 10% menor

m=1.0e-3 / 6.022e23  # Masa del ion
q=1.6e-19   #Carga del ion
H=0.2       # 2*H es la distancia entre las placas horizontales del selector
dV=25     # Diferencia de potencial entre ambas placas
#dVr=(np.random.rand())*10+20
#dV=round(dVr,2)
E=dV/2./H   # Campo eléctrico uniforme entre las placas

B=8.21e-4  # Campo magnético
L=2.0  # Distancia horizontal del selector

Ec=30*1.6e-19  # Energía cinética del ion que entra en el detector
#Ecr=(np.random.rand())*40+10
#Ec=round(Ecr,2)
vx0=np.sqrt(2*Ec/m)  #Velocidad del ion que entra al selector
dy=0.05 * H # 2*dy es el tamaño de la apertura que hay en la salida del detector

tf=10.0   #tiempo final de simulacion


qm=q/m
par=[qm,E,B,L,H,dy]


# Definicion de las ecuaciones de movimiento de los iones

def selector(z,t,par):
    x,vx,y,vy=z


    if x < L and x > 0:
      cB=1.
      cE=1.
    else:
      cB=0.
      cE=0.

    yy=math.fabs(y)

    if yy > H :
        vx=0
        vy=0

    if x > L and yy>dy:
        vx=0
        vy=0

    
    dzdt=[vx,qm*vy*cB*B,vy,qm*cE*E-qm*vx*cB*B]
    return dzdt



def selector2(z2,t,par):
    x2,vx2,y2,vy2=z2


    if x2 < L and x2 > 0:
      cB2=1.
      cE2=1.
    else:
      cB2=0.
      cE2=0.

    yy2=math.fabs(y2)

    if yy2 > H :
        vx2=0
        vy2=0

    if x2 > L and yy2>dy:
        vx2=0
        vy2=0

    
    dzdt2=[vx2,qm*vy2*cB2*B,vy2,qm*cE2*E-qm*vx2*cB2*B]
    return dzdt2


def selector3(z3,t,par):
    x3,vx3,y3,vy3=z3


    if x3 < L and x3 > 0:
      cB3=1.
      cE3=1.
    else:
      cB3=0.
      cE3=0.

    yy3=math.fabs(y3)

    if yy3 > H :
        vx3=0
        vy3=0

    if x3 > L and yy3>dy:
        vx3=0
        vy3=0

    
    dzdt3=[vx3,qm*vy3*cB3*B,vy3,qm*cE3*E-qm*vx3*cB3*B]
    return dzdt3





#Llamada a la rutina odeint que resuelve las ecuaciones de movimiento

nt=10000
z0=[-0.1,vx0,0.0,0.0]   
z02=[-0.1,1.1*vx0,0.0,0.0]
z03=[-0.1,0.9*vx0,0.0,0.0] 

  
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z=odeint(selector,z0,t,args=(par,),atol=abserr, rtol=relerr)
z2=odeint(selector,z02,t,args=(par,),atol=abserr, rtol=relerr)
z3=odeint(selector,z03,t,args=(par,),atol=abserr, rtol=relerr)



# Definicion del grafico 
# Cambiar los limites de acuerdo a lo que se necesite
# asi como los titulos de los ejes y los labels

ax = plt.axes(xlim=(-0.1,L+0.1), ylim=(-0.02-H,H+0.02))


line1, = ax.plot(z[:,0],z[:,2], linewidth=1,
                 label='v=v0')

line2, = ax.plot(z2[:,0],z2[:,2], linewidth=1,
                 label='v=1.1*v0')

line3, = ax.plot(z3[:,0],z3[:,2], linewidth=1,
                 label='v=0.9*v0')


fontP = FontProperties()
fontP.set_size('small')

ax.legend(loc=0, ncol=1, bbox_to_anchor=(0.5, 0.6),
            prop = fontP,fancybox=True,shadow=False)

ax.set_ylabel("y [m]")
ax.set_xlabel("x [m]")





line,=ax.plot([],[],lw=2)


# Crea rectangulos, flechas en el dibujo
rect = patches.Rectangle((0,-H),L,H*2,linewidth=2,edgecolor='r',facecolor='none')
rect2 = patches.Rectangle((-dy,-dy),3*dy,2*dy,linewidth=2,edgecolor='none',facecolor='w')
rect3 = patches.Rectangle((L-dy,-dy),3*dy,2*dy,linewidth=2,edgecolor='none',facecolor='w')

arrow= patches.Arrow(0.1*L,0.5*H,0.0,0.2*H,width=0.01)
arrow2= patches.Arrow(0.15*L,0.5*H,0.0,0.2*H,width=0.01)

circle1=patches.Circle((0.1*L,-0.5*H),0.1*H,fill=False)
circle2=patches.Circle((0.1*L,-0.5*H),0.02*H)
            

# agrega los eleementos anteriores al dibujo
ax.add_patch(rect)
ax.add_patch(rect2)
ax.add_patch(rect3)
ax.add_patch(arrow)
ax.add_patch(arrow2)
ax.add_patch(circle1)
ax.add_patch(circle2)
ax.text(0.2*L,-0.55*H, "B",fontsize=20)
ax.text(0.2*L, 0.52*H, "E", fontsize=20)


plt.show()
