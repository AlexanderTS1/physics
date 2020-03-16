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
Ejercicio 6 
==============================================================================

#============================================================================== 
#Chebyshev
#==============================================================================

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

# No devuelve polinomios mónicos
p5=Chebyshev(5)
  
# Lo podemos hacer mónico:
p5/p5[0]


def Chebyshev_hasta(n):
    T=[]
    for i in range(n+1):
        T.append(Chebyshev(i))    
    return np.array(T)

Chebyshev_hasta(6)

# Con la clase Chebyshev de Python:

T5Ch = np.polynomial.chebyshev.Chebyshev([0,0,0,0,0,1])
nodCheby5=T5Ch.roots()
T5Ch=npol.polyfromroots(nodCheby5)
T5Ch=npol.polyfromroots(nodCheby5)[::-1]


#============================================================================== 
#Legendre
#==============================================================================

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

def Legendre_hasta(n):
    T=[]
    for i in range(n+1):
        T.append(Legendre(i))    
    return np.array(T)

# Con la clase Legendre de Python:
T5L = np.polynomial.legendre.Legendre([0,0,0,0,0,1])
nodLeg5=T5L.roots()
T5L=npol.polyfromroots(nodLeg5)
T5L=npol.polyfromroots(nodLeg5)[::-1]


==============================================================================
Ejercicio 7
==============================================================================

def fun1(x):
    return np.exp(-2*x)*np.cos(3*x)



#============================================================================== 
# Ejercicio 7a
#==============================================================================

def wCh(x):
    return 1/np.sqrt(1-x**2)

from scipy.integrate import quad


a0Ch=quad(lambda x:fun1(x)*wCh(x),-1,1)[0]/quad(wCh,-1,1)[0]
a1Ch=quad(lambda x:fun1(x)*wCh(x)*x,-1,1)[0]/quad(lambda x:wCh(x)*x**2,-1,1)[0]
a2Ch=quad(lambda x:fun1(x)*wCh(x)*(x**2-0.5),-1,1)[0]/quad(lambda x:wCh(x)*(x**2-0.5)**2,-1,1)[0]

def aproxCheb(x):
    return a0Ch+a1Ch*x+a2Ch*(x**2-0.5)

xreal=np.linspace(-1,1)
plt.plot(xreal,fun1(xreal),'k',xreal,aproxCheb(xreal),'r')
plt.legend(('Función','Aprox. Chebyshev 2'),loc = 'best')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')


#============================================================================== 
#Legendre
#==============================================================================

a0L=quad(fun1,-1,1)[0]/2
a1L=quad(lambda x:fun1(x)*x,-1,1)[0]/quad(lambda x:x**2,-1,1)[0]
a2L=quad(lambda x:fun1(x)*(x**2-0.3333),-1,1)[0]/quad(lambda x:(x**2-0.3333)**2,-1,1)[0]

def aproxLeg(x):
    return a0L+a1L*x+a2L*(x**2-0.3333)

xreal=np.linspace(-1,1)
plt.plot(xreal,fun1(xreal),'k',xreal,aproxLeg(xreal),'r')
plt.legend(('Funcion','Aprox. Legendre 2'),loc = 'best')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')



#============================================================================== 
# Ejercicio 7b
#==============================================================================

plt.plot(xreal,fun1(xreal),'k',xreal,aproxCheb(xreal),'g',xreal,aproxLeg(xreal),'r')
plt.legend(('Funcion','Aprox. Chebyshev 2','Aprox. Legendre 2'),loc = 'best')
plt.xlabel('Abscisas')
plt.ylabel('Ordenadas')


#============================================================================== 
# Ejercicio 7c
#==============================================================================


errorCheb=(quad(lambda x:wCh(x)*(fun1(x)-aproxCheb(x))**2,-1,1)[0])**0.5
errorLeg=(quad(lambda x:(fun1(x)-aproxLeg(x))**2,-1,1)[0])**0.5
