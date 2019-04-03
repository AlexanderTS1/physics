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

m=1.673e-27  # masa del ion
q=1.602e-19   # Carga del ion
B=1.0  # Campo magnético
#Ecr=(np.random.rand())*40+10
#Ec=round(Ecr,2)
vx0=1.0e2  # velocidad del ion
tf=12e-8  #tiempo de simulacion. Debe ser 2 o 3 veces superior al periodo del ion
# para poder ver varios ciclos


c=3e8  #velocidad de la luz en el vacio
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


# Definicion del grafico
# Modificar los limites de acuerdo a las necesidades, así como los titulos de los ejes

matplotlib.rc('xtick', labelsize=32) 
matplotlib.rc('ytick', labelsize=32) 
plt.rc('text', usetex=False)
plt.rc('font', family='serif')
plt.rc('font', size='36')

ax=plt.axes(xlim=(0,tf))


# Llamada a odeint que resuleve las ecuaciones de movimiento

nt=10000
z0=[-0.1,vx0,0.,0.0]    
t=np.linspace(0,tf,nt)
abserr = 1.0e-8
relerr = 1.0e-6
z=odeint(circulo,z0,t,args=(par,),atol=abserr, rtol=relerr)

min_ = min(z[:,2])
print("El mínimo es " + str(min_))

minimums_ = []
minimums_t_ = []
for i in range(len(t)):
    if abs(z[i, 2] - min_) < 1.0e-10:
        minimums_.append(z[i, 2])
        minimums_t_.append(t[i])

print("mins = " + str(minimums_))
print("t = " + str(minimums_t_))

z_0_ = min(minimums_)
t_0_ = min(minimums_t_)
z_1_ = max(minimums_)
t_1_ = max(minimums_t_)

print(z_0_)
print(t_0_)
print(z_1_)
print(t_1_)

T = t_1_ - t_0_

print("T = " + str(T))

line1, = ax.plot(t[:],z[:,2],'-', label=r"v = " + str(vx0/1e8) + r" $\cdot 10^8 [ms^{-1}]$", linewidth=2)
ax.plot([t_0_, t_1_], [z_0_, z_1_], 'ro-')
ax.annotate("T = " + "{:.3e}".format(T), (t_0_*1.2, z_0_*0.95))
ax.legend(loc='upper right')

ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))

ax.set_ylabel("y [m]")
ax.set_xlabel("t [s]")






plt.show()
