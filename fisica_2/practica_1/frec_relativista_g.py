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


# Este programa calcula la trayectoria circular de un ion en presencia de un
# Campo magnético perpendicular a su velocidad y luego representa
# la coordenada y del ion en funcion del tiempo


# Datos a modificar 

m=1.  # masa del ion
q=1.   # Carga del ion
B=1.  # Campo magnético
#Ecr=(np.random.rand())*40+10
#Ec=round(Ecr,2)
vx0=10  # velocidad del ion
tf=20.0  #tiempo de simulacion. Debe ser 2 o 3 veces superior al periodo del ion
# para poder ver varios ciclos


c=10000.0  #velocidad de la luz en el vacio
g=np.sqrt(1-((vx0/c)*(vx0/c))) #factor gamma
mr=m*g  # si se quiere estudiar el caso relativista usar mr=m*g si no mr=m 
#mr=m



qm=q/mr
par=[qm,B]


# Definicion de las ecuaciones de movimiento
def circulo(z,t,par):
    x,vx,y,vy=z
    cB=1.  
    dzdt=[vx,qm*vy*cB*B,vy,-qm*vx*cB*B]
    return dzdt


# Llamada a odeint que resuleve las ecuaciones de movimiento

nt=10000
z0=[-0.1,vx0,0.,0.0]    
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z=odeint(circulo,z0,t,args=(par,),atol=abserr, rtol=relerr)



# Definicion del grafico
# Modificar los limites de acuerdo a las necesidades, así como los titulos de los ejes

ax=plt.axes(xlim=(0,tf))


line1, = ax.plot(t[:],z[:,2],'--', linewidth=2)

ax.legend(loc='lower right')

ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

ax.set_ylabel("y")
ax.set_xlabel("t")






plt.show()
