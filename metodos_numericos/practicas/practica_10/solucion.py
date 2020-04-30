# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:58:00 2020

@author: Julio Mulero & José Vicente
"""

###################################################
import numpy as np
import matplotlib.pyplot as plt
###################################################


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 1
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def busqueda_incremental(f,a,b,n):
    # f: funcion que determina la ecuación
    # a: extremo inferior del intervalo
    # b: extremo superior del intervalo
    # n: número de subintervalos
    extremos=np.linspace(a,b,n+1)
    intervalos=np.zeros((n,2))
    lista=[]
    for i in range(n):
        intervalos[i,0]=extremos[i]
        intervalos[i,1]=extremos[i+1]
        if f(extremos[i])*f(extremos[i+1])<=0:
            lista.append(i)
    return intervalos[lista,::]

# Un ejemplo:
    
def fun1(x):
    return np.exp(-x/10)*np.cos(x)

busqueda_incremental(fun1,0,15,50)
x = np.linspace(0,15,150)
plt.plot(x,fun1(x),x,0*x)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 2
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def biseccion(f,a,b,tol):
    # f: funcion que determina la ecuación
    # a: extremo inferior del intervalo
    # b: extremo superior del intervalo
    # tol: tolerancia
    i = 0
    while abs(b-a)>=tol:
        p = (a+b)/2.0
        if f(p) == 0:
            return [p,i]
        else:
            if f(a)*f(p)>0:
                a = p
            else:
                b = p
        i = i+1
    return [p,i]

# Un ejemplo:
    
def fun2(x):
    return np.exp(-x/10)*np.cos(x)

tol = 10**(-4)

biseccion(fun2,1.5,1.8,tol)
biseccion(fun2,4.5,4.8,tol)
biseccion(fun2,7.8,8.1,tol)
biseccion(fun2,10.8,11.1,tol)
biseccion(fun2,14.1,14.4,tol)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 3
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def punto_fijo(g,p0,tol,maxiter):
    # g: funcion que determina la ecuación
    # p0: punto inicial
    # tol: tolerancia 
    # maxiter: máximo número de iteraciones permitidas
    for i in range(maxiter):      
        p1 = g(p0)
        if abs(p1-p0)<tol:
            return [p1,i]
        p0 = p1
    print('Número máximo de iteraciones alcanzado!')
    return [None,None]
    
# Un ejemplo:
    
def fun3(x):
    return np.cos(x)-x*np.exp(x)

fun3(0)

# Por ejemplo, podríamos considerar:
    
def gfun3(x):
    return np.cos(x)/np.exp(x)

tol = 10**(-4)

sol = punto_fijo(gfun3,2,tol,50)

fun3(sol[0])

print('Iteraciones:',sol[1])


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 4
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def newton(f,df,p0,tol,maxiter):
    # f: funcion que determina la ecuación
    # df: derivada de f
    # p0: punto inicial
    # tol: tolerancia 
    # maxiter: máximo número de iteraciones permitidas
    for i in range(maxiter):      
        p1 = p0-f(p0)/df(p0)
        if abs(p1-p0)<tol:
            return [p1,i]
        p0 = p1
    print('Número máximo de iteraciones alcanzado!')
    return [None,None]
    
# Un ejemplo:
    
def fun3(x):
    return np.exp(-x/10)*np.cos(x)

def dfun3(x):
    return -np.exp(-x/10)*np.cos(x)/10-np.exp(-x/10)*np.sin(x)

tol = 10**(-4)

newton(fun3,dfun3,1.65,tol,50)
newton(fun3,dfun3,4.65,tol,50)
newton(fun3,dfun3,7.95,tol,50)
newton(fun3,dfun3,10.95,tol,50)
newton(fun3,dfun3,14.25,tol,50)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 5
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    
def secante(f,p0,p1,tol,maxiter):
    # f: funcion que determina la ecuación
    # p0 y p1: puntos iniciales
    # tol: tolerancia 
    # maxiter: máximo número de iteraciones permitidas
    p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0))
    i = 1
    while (i<maxiter) and abs(p2-p1)>=tol: 
        i=i+1
        p0 = p1
        p1 = p2
        p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0))
    return [p2,i]

# Un ejemplo:
    
def fun4(x):
    return np.exp(-x/10)*np.cos(x)

tol = 10**(-4)

secante(fun4,1.5,1.8,tol,50)
secante(fun4,2.5,4.8,tol,50)
secante(fun4,7.8,8.1,tol,50)
secante(fun4,10.8,11.1,tol,50)
secante(fun4,14.1,14.4,tol,50)

def regula_falsi(f,p0,p1,tol,maxiter):
    # f: funcion que determina la ecuación
    # p0 y p1: puntos iniciales
    # tol: tolerancia 
    # maxiter: máximo número de iteraciones permitidas
    p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0)) #Primer paso
    i = 1
    while  (i<maxiter) and abs(p2-p1)>=tol:
        i = i+1
        if f(p1)*f(p2)<0:
            p0 = p1
        p1 = p2
        p2 = p1-((p1-p0)*f(p1))/(f(p1)-f(p0)) 
    return [p2 ,i]

# Un ejemplo:
    
def fun5(x):
    return np.exp(-x/10)*np.cos(x)

tol = 10**(-4)

regula_falsi(fun5,1.5,1.8,tol,50)
regula_falsi(fun5,4.5,4.8,tol,50)
regula_falsi(fun5,7.8,8.1,tol,50)
regula_falsi(fun5,10.8,11.1,tol,50)
regula_falsi(fun5,14.1,14.4,tol,50)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 6
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# La distancia entre los puntos (2cos(t),sin(t),0) y (2,1,0) viene dada por:
# d(t)=sqrt((2cos(t)-2)^2+(sin(t)-1)^2+0))
# Buscamos el mínimo de dicha función, por tanto, la ecuación a resolver d'(t)=0

# Ahora bien, minimizar una raíz cuadrada equivale a minimizar su argumento:

def dist(x):
    return (2.*np.cos(x)-2)**2+(np.sin(x)-1)**2

x = np.linspace(0,2)
plt.plot(x,dist(x))

def ddist(x):
    return 2*(2*np.cos(x)-2)*(-2*np.sin(x))+2*(np.sin(x)-1)*(np.cos(x))

x = np.linspace(0,2)
plt.plot(x,ddist(x),x,0*x)

# La ecuación a resolver es ddist(x)=0. Necesitamos la derivada de ddist:
    
def dddist(x):
    a = 2*(-2*np.sin(x))**2+2*(2*np.cos(x)-2)*(-2*np.cos(x))
    b =2*np.cos(x)+2*(np.sin(x)-1)*np.sin(x)
    return a+b

# La solución de la ecuación no lineal es el momento de tiempo en que la distancia 
# es mínima:
    
tol = 10**(-4)
busqueda_incremental(ddist,0,2,10)

# Con el método de la bisección:
    
sol_biseccion = biseccion(ddist,0.4,0.6,tol)
print('Solución:',sol_biseccion) 

# Con el método de Newton:
    
sol_newton = newton(ddist,dddist,0.5,tol,50)
print('Solución:',sol_newton) 

# Con el método de la secante:

sol_secante = secante(ddist,0.4,0.6,tol,50)
print('Solución:',sol_secante) 

# Con el método de regula-falsi:

sol_regulafalsi = regula_falsi(ddist,0.4,0.6,tol,50)
print('Solución:',sol_regulafalsi) 

# La distancia será:

np.sqrt(dist(sol_newton[0]))

