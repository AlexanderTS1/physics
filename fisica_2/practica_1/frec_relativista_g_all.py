import numpy as np
import math
import random
import matplotlib
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

m=1.67e-27  # masa del ion
q=1.6e-19   # Carga del ion
B=1.0  # Campo magnético
#Ecr=(np.random.rand())*40+10
#Ec=round(Ecr,2)
vx0=[0.2e8, 0.6e8, 0.8e8, 1.0e8]  # velocidad del ion
tf=12e-8  #tiempo de simulacion. Debe ser 2 o 3 veces superior al periodo del ion
# para poder ver varios ciclos


c=3e8  #velocidad de la luz en el vacio



# Definicion de las ecuaciones de movimiento
def circulo(z,t,par):
    x,vx,y,vy=z
    cB=1.  
    dzdt=[vx,qm*vy*cB*B,vy,-qm*vx*cB*B]
    return dzdt


# Definicion del grafico
# Modificar los limites de acuerdo a las necesidades, así como los titulos de los ejes

matplotlib.rc('xtick', labelsize=24) 
matplotlib.rc('ytick', labelsize=24) 
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size='32')

ax=plt.axes(xlim=(0,tf))


# Llamada a odeint que resuleve las ecuaciones de movimiento

for v in range(len(vx0)):

    g=np.sqrt(1-((vx0[v]/c)*(vx0[v]/c))) #factor gamma
    mr=m*g  # si se quiere estudiar el caso relativista usar mr=m*g si no mr=m 
    #mr=m
    qm=q/mr
    par=[qm,B]

    nt=10000
    z0=[-0.1,vx0[v],0.,0.0]    
    t=np.linspace(0,tf,nt)
    abserr = 1.0e-8
    relerr = 1.0e-6
    z=odeint(circulo,z0,t,args=(par,),atol=abserr, rtol=relerr)

    min_ = 0.0
    last_t_ = 0.0

    for i in range(len(t[:])):

        if (abs(z[i, 2] - min_) < 1e-12):
            print("Encontrado siguiente mínimo")
            print("min = " + str(min_))
            print("z = " + str(z[i, 2]))
            print("En t = " + str(t[i]))

        if (z[i, 2] < min_):
            min_ = z[i, 2]
            last_t_ = t[i]

    line1, = ax.plot(t[:],z[:,2],'-', label=r"v = " + str(vx0[v]/1e8) + r" $\cdot 10^8 [ms^{-1}]$", linewidth=2)

ax.legend(loc='lower center')

ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

ax.set_ylabel("y [m]")
ax.set_xlabel("t [s]")






plt.show()
