# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 18:00:33 2020

@author: Julio Mulero
"""

###################################################
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import numpy.polynomial.polynomial as npol
from scipy.integrate import quad
###################################################

# POLINOMIO INTERPOLADOR DE LAGRANGE

def PolLagrange(x,y):
    return np.dot(np.linalg.inv(np.vander(x)),y)

# DIFERENCIAS DIVIDIDAS (como matriz triangular inferior)

def dif_divididas(x,y):
    A = np.zeros((len(x),len(x)))
    for i in range(len(x)):    
        A[i,0] = y[i]
    for j in range(1,len(x)):
        for i in range(j,len(x)):
            A[i,j] = (A[i][j-1]-A[i-1][j-1])/(x[i]-x[i-j])   
    return A 
    
# POLINOMIO INTERPOLADOR DE NEWTON  

def PolNewton(x,y):
    
    DD = np.diag(dif_divididas(x,y))
    polinomio = np.zeros(len(x))
    for i in range(len(x)):
        polinomio = np.polyadd(polinomio,DD[i]*npol.polyfromroots(x[:i])[::-1])
    return polinomio
  
==============================================================================
Ejercicio 1
==============================================================================

def fun1(x):
    return np.cos(np.arctan(x))-np.exp(x**2)*np.log(x+2)

xreal1=np.linspace(-1,1)
plt.plot(xreal1,fun1(xreal1),'k')

# ====================================
#  Ejercicio 1a
# ====================================

def Chebyshev(n):
    if n == 0:
        T = np.array([1.])
        return T
    elif n == 1:
        T = np.array([1.,0])
        return T
    else:
        Tn1 = Chebyshev(n-1)
        Tn2 = Chebyshev(n-2)
        Tn = np.polysub(np.polymul(np.array([2.,0]),Tn1),Tn2)    
    return Tn

p5Ch=Chebyshev(5)
raices5Ch=np.roots(p5Ch)

P_Newton=PolNewton(raices5Ch,fun1(raices5Ch))

# ====================================
#  Ejercicio 1b
# ====================================

from scipy.integrate import quad

p0Ch=Chebyshev(0)
p1Ch=Chebyshev(1)
p2Ch=Chebyshev(2)
p3Ch=Chebyshev(3)
p4Ch=Chebyshev(4)

p0Ch=p0Ch/p0Ch[0]
p1Ch=p1Ch/p1Ch[0]
p2Ch=p2Ch/p2Ch[0]
p3Ch=p3Ch/p3Ch[0]
p4Ch=p4Ch/p4Ch[0]

def wCh(x):
    return 1/np.sqrt(1-x**2)

a0Ch=quad(lambda x:fun1(x)*wCh(x),-1,1)[0]/quad(wCh,-1,1)[0]
a1Ch=quad(lambda x:fun1(x)*wCh(x)*x,-1,1)[0]/quad(lambda x:wCh(x)*x**2,-1,1)[0]
a2Ch=quad(lambda x:fun1(x)*wCh(x)*(x**2-0.5),-1,1)[0]/quad(lambda x:wCh(x)*(x**2-0.5)**2,-1,1)[0]
a3Ch=quad(lambda x:fun1(x)*wCh(x)*(x**3-0.75*x),-1,1)[0]/quad(lambda x:wCh(x)*(x**3-0.75*x)**2,-1,1)[0]
a4Ch=quad(lambda x:fun1(x)*wCh(x)*(x**4-x**2+0.125),-1,1)[0]/quad(lambda x:wCh(x)*(x**4-x**2+0.125)**2,-1,1)[0]

def aproxCheb(x):
    return a0Ch+a1Ch*x+a2Ch*(x**2-0.5)+a3Ch*(x**3-0.75*x)+a4Ch*(x**4-x**2+0.125)

# ====================================
#  Ejercicio 1c
# ====================================

xreal=np.linspace(-1,1)
plt.plot(xreal,fun1(xreal),xreal,aproxCheb(xreal),'k',xreal,aproxCheb(xreal),'r')
plt.legend(('Función','Pol. Newton','Aprox. Chebyshev 4'),loc = 'best')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')

==============================================================================
Ejercicio 2
==============================================================================

def ones_to_ab(y,a,b):
    return a+(y+1)*(b-a)/2.

def ab_to_ones(x,a,b):
    return -1+2*(x-a)/(b-a)

ab_to_ones(4.5,3,6)
ones_to_ab(0,3,6)


# Chebyshev:
     
raices0Ch=np.roots(p0Ch)
raices1Ch=np.roots(p1Ch)
raices2Ch=np.roots(p2Ch)
raices3Ch=np.roots(p3Ch)

T0Chtras=npol.polyfromroots(ones_to_ab(raices0Ch,3,6))[::-1]
T1Chtras=npol.polyfromroots(ones_to_ab(raices1Ch,3,6))[::-1]
T2Chtras=npol.polyfromroots(ones_to_ab(raices2Ch,3,6))[::-1]
T3Chtras=npol.polyfromroots(ones_to_ab(raices3Ch,3,6))[::-1]

#La función peso para la aproximación con los polinomios de Tchebyshev es:
def wChtras(x):
    return 1/np.sqrt(1-ab_to_ones(x,3,6)**2)

quad(lambda x:wChtras(x)*np.polyval(T1Chtras,x)*np.polyval(T3Chtras,x),3,6)[0]


T3Ch=npol.polyfromroots(raices3Ch)[::-1]
xreal31=np.linspace(-1,1)
plt.plot(xreal31,np.polyval(T3Ch,xreal31),'k')
xreal32=np.linspace(3,6)
plt.plot(xreal32,np.polyval(T3Chtras,xreal32),'k')

# A partir de la clase Chebyshev de Python:

T3Chtras = np.polynomial.chebyshev.Chebyshev([0,0,0,1],[3,6])
nodCheby3=T3Chtras.roots()
T3Chtras=npol.polyfromroots(nodCheby3) # ¡ Potencias crecientes !
T3Chtras=npol.polyfromroots(nodCheby3)[::-1] # ¡ Potencias decrecientes !

# Legendre:

def Legendre(n):
     if n == 0:
        T = np.array([1.])
        return T
     elif n == 1:
        T = np.array([1.,0])
        return T
     else:
        Tn1 = Legendre(n-1)
        Tn2 = Legendre(n-2)
        Tn = np.polysub(np.polymul(np.array([(2.*(n-1) + 1.)/(n),0]),Tn1),np.polymul(np.array([(n-1.)/n]),Tn2))   
     return Tn

p0L=Legendre(0)
p1L=Legendre(1)
p2L=Legendre(2)
p3L=Legendre(3)

raices0L=np.roots(p0L)
raices1L=np.roots(p1L)
raices2L=np.roots(p2L)
raices3L=np.roots(p3L)

T0Ltras=npol.polyfromroots(ones_to_ab(raices0L,3,6))[::-1]
T1Ltras=npol.polyfromroots(ones_to_ab(raices1L,3,6))[::-1]
T2Ltras=npol.polyfromroots(ones_to_ab(raices2L,3,6))[::-1]
T3Ltras=npol.polyfromroots(ones_to_ab(raices3L,3,6))[::-1]

quad(lambda x:np.polyval(T1Ltras,x)*np.polyval(T2Ltras,x),3,6)[0]

# A partir de la clase Legendre de Python:

T3Ltras = np.polynomial.legendre.Legendre([0,0,0,1],[3,6])
nodLeg3 = T3Ltras.roots()
T3Ltras = npol.polyfromroots(nodLeg3) # ¡ Potencias crecientes !
T3Ltras = npol.polyfromroots(nodLeg3)[::-1] # ¡ Potencias decrecientes !

==============================================================================
Ejercicio 3
==============================================================================

def fun3(x):
    return np.exp(-x)*np.cos(2*x)

xreal3=np.linspace(3,6)
plt.plot(xreal3,fun3(xreal3),'k')

# ====================================
#  Ejercicio 3a
# ====================================

def wChtras(x):
    return 1/np.sqrt(1-ab_to_ones(x,3,6)**2)

xreal31=np.linspace(-1,1)
plt.plot(xreal31,wCh(xreal31),'k')
xreal32=np.linspace(3,6)
plt.plot(xreal32,wChtras(xreal32),'k')

a0Ch=quad(lambda x:wChtras(x)*fun3(x),3,6)[0]/quad(lambda x:wChtras(x),3,6)[0]
a1Ch=quad(lambda x:wChtras(x)*fun3(x)*(x-4.5),3,6)[0]/quad(lambda x:wChtras(x)*(x-4.5)**2,3,6)[0]
a2Ch=quad(lambda x:wChtras(x)*fun3(x)*(x**2-9*x+19.125),3,6)[0]/quad(lambda x:wChtras(x)*(x**2-9*x+19.125)**2,3,6)[0]
a3Ch=quad(lambda x:wChtras(x)*fun3(x)*(x**3-13.5*x**2+59.0625*x-83.53125),3,6)[0]/quad(lambda x:wChtras(x)*(x**3-13.5*x**2+59.0625*x-83.53125)**2,3,6)[0]

def aproxCheb(x):
    return a0Ch+a1Ch*(x-4.5)+a2Ch*(x**2-9*x+19.125)+a3Ch*(x**3-13.5*x**2+59.0625*x-83.53125)

xreal=np.linspace(3,6)
plt.plot(xreal,fun3(xreal),xreal,aproxCheb(xreal))

# ====================================
#  Ejercicio 3b
# ====================================

a0L=quad(fun3,3,6)[0]/3.
a1L=quad(lambda x:fun3(x)*(x-4.5),3,6)[0]/quad(lambda x:(x-4.5)**2,3,6)[0]
a2L=quad(lambda x:fun3(x)*(x**2-9*x+19.5),3,6)[0]/quad(lambda x:(x**2-9*x+19.5)**2,3,6)[0]
a3L=quad(lambda x:fun3(x)*(x**3-13.5*x**2+59.4*x-85.05),3,6)[0]/quad(lambda x:(x**3-13.5*x**2+59.4*x-85.05)**2,3,6)[0]

def aproxLeg(x):
    return a0L+a1L*(x-4.5)+a2L*(x**2-9*x+19.5)+a3L*(x**3-13.5*x**2+59.4*x-85.05)

# ====================================
#  Ejercicio 3c
# ====================================

xreal=np.linspace(3,6)
plt.plot(xreal,fun3(xreal),'k',xreal,aproxCheb(xreal),xreal,aproxLeg(xreal))
plt.legend(('Funcion','Aprox. Chebyshev 3','Aprox. Legendre 3'),loc = 'best')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')

