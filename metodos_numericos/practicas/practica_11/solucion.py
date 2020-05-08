# -*- coding: utf-8 -*-
"""
PRÁCTICA 11 - 08/05/2020

@authors: Julio Mulero  &  José Vicente  
"""

###################################################
import numpy as np
import matplotlib.pyplot as plt
###################################################

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 1
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

''' INPUT DE LA FUNCIÓN "punto_fijo_sist"

    G: función vectorial que determina el sistema
    p0: punto inicial (semilla)
    tol: tolerancia
    maxiter: número máximo de iteraciones permitidas '''
    
def punto_fijo_sist(G,p0,tol,maxiter):
    for i in range(maxiter):      
        p1 = G(p0)
        if np.linalg.norm(p1-p0)<tol:
            return [p1,i+1]
        p0 = p1
    print('Número máximo de iteraciones alcanzado!')
    return [None,None]

# En este caso, como trabajamos con vectores, hemos de usar la norma en lugar
# del valor absoluto que se utilizaba con escalares.
    
def G1(w):
    x,y = w
    return np.array([-0.5*x**2+2*x+0.5*y-0.25, -0.125*x**2-0.5*y**2+y+0.5])

tol = 10**(-4)
maxiter = 50

p0 = np.array([0,1])
punto_fijo_sist(G1,p0,tol,maxiter)
    # [array([1.90067759, 0.31125858]), 19]

p0 = np.array([2,0])
punto_fijo_sist(G1,p0,tol,maxiter)
    # [array([1.9006801 , 0.31119331]), 17]


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 2
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

''' INPUT DE LA FUNCIÓN "newton_sist"

    F: función vectorial que determina el sistema
    JF: matriz Jacobiana
    p0: punto inicial (semilla)
    tol: tolerancia
    itermax: número máximo de iteraciones permitidas '''

def newton_sist(F,JF,p0,tol,maxiter):
    for i in range(maxiter):
        y = np.linalg.solve(JF(p0),-F(p0))
        p1 = p0 + y
        if np.linalg.norm(p1-p0)<tol:
            return [p1,i+1]
        p0 = p1
    print('Número máximo de iteraciones alcanzado!')
    return [None,None]
    
#  (a)
      
def F1a(w):
    x,y = w
    return np.array([3*x**2-y**2, 3*x*y**2-x**3-1])

def JF1a(w):
    x,y = w
    return np.array([ [6*x , -2*y], [3*y**2-3*x**2 , 6*x*y]])

tol = 10**(-4)
maxiter = 50

p0a = np.array([1,1])
S1Na = newton_sist(F1a,JF1a,p0a,tol,maxiter)
print('Solución encontrada: p =',S1Na[0],' en ',S1Na[1],' iteraciones')
    # Solución encontrada: p = [0.5       0.8660254]  en  4  iteraciones
p0a = np.array([-1,-1])
S1Na = newton_sist(F1a,JF1a,p0a,tol,maxiter)
print('Solución encontrada: p =',S1Na[0],' en ',S1Na[1],' iteraciones')
    # Solución encontrada: p = [ 0.5       -0.8660254]  en  9  iteraciones


#  (b)
      
def F1b(w):
    x,y = w
    return np.array([-np.exp(x**2)+8*x*np.sin(y), x+y-1])

def JF1b(w):
    x,y = w
    return np.array([ [-np.exp(x**2)*2*x+8*np.sin(y) , 8*x*np.cos(y)], [1 , 1]])
    
tol = 10**(-4)
maxiter = 50

p0b = np.array([2,2])
S1Nb = newton_sist(F1b,JF1b,p0b,tol,maxiter)
print('Solución encontrada: p =',S1Nb[0],' en ',S1Nb[1],' iteraciones')
    # Solución encontrada: p = [0.70424697 0.29575303]  en  9  iteraciones
    
p0b = np.array([0,1])
S1Nb = newton_sist(F1b,JF1b,p0b,tol,maxiter)
print('Solución encontrada: p =',S1Nb[0],' en ',S1Nb[1],' iteraciones')
    # Solución encontrada: p = [0.17559892 0.82440108]  en  4  iteraciones


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 3
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# Calculamos en primer lugar la aproximación del Jacobiano:
    
def JF_approx(F,p0,h):
    n = len(p0)
    JFa = np.zeros((n,n))
    for i in range(n):
        v = np.eye(n)[i]
        JFa[:,i] = (F(p0+h*v)-F(p0))/(h)
    return JFa

''' INPUT DE LA FUNCIÓN "newton_approx"

    F: función vectorial que determina el sistema
    p0: punto inicial (semilla)
    tol: tolerancia
    itermax: número máximo de iteraciones permitidas '''

def newton_approx(F,p0,tol,maxiter):
    for i in range(maxiter):
        JFa = JF_approx(F,p0,10**(-2))
        y = np.linalg.solve(JFa,-F(p0))
        p1 = p0 + y
        if np.linalg.norm(p1-p0)<tol:
            return [p1,i+1]
        p0 = p1
    print('Número máximo de iteraciones alcanzado!')
    return [None,None]
    
#  (a)
      
def F1a(w):
    x,y = w
    return np.array([3*x**2-y**2, 3*x*y**2-x**3-1])

tol = 10**(-4)
maxiter = 50

p0a = np.array([1,1])
S1Na = newton_approx(F1a,p0a,tol,50)
print('Solución encontrada: p =',S1Na[0],' en ',S1Na[1],' iteraciones')
    # Solución encontrada: p = [0.5       0.8660254]  en  5  iteraciones
    
p0a = np.array([-1,-1])
S1Na = newton_approx(F1a,p0a,tol,50)
print('Solución encontrada: p =',S1Na[0],' en ',S1Na[1],' iteraciones')
    # Solución encontrada: p = [ 0.49999999 -0.86602541]  en  10  iteraciones

# REPRESENTACIÓN GRÁFICA:

xd = np.linspace(-2,2,100)
yd = np.linspace(-2,2,100)
C = [0]
X, Y = np.meshgrid(xd, yd)
plt.figure()
plt.contour(X,Y,F1a([X,Y])[0],C,colors='red')
plt.contour(X,Y,F1a([X,Y])[1],C,colors='blue')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.grid(True)
plt.show()


#  (b)
      
def F1b(w):
    x,y = w
    return np.array([-np.exp(x**2)+8*x*np.sin(y), x+y-1])

tol = 10**(-4)
maxiter = 50

p0b = np.array([2,2])
S1Nb = newton_approx(F1b,p0b,tol,maxiter)
print('Solución encontrada: p =',S1Nb[0],' en ',S1Nb[1],' iteraciones')
    # Solución encontrada: p = [0.70424697 0.29575303]  en  9  iteraciones
    
p0b = np.array([0,1])
S1Nb = newton_approx(F1b,p0b,tol,maxiter)
print('Solución encontrada: p =',S1Nb[0],' en ',S1Nb[1],' iteraciones')
    # Solución encontrada: p = [0.17559892 0.82440108]  en  4  iteraciones

# REPRESENTACIÓN GRÁFICA:

xd = np.linspace(-2,3,100)
yd = np.linspace(-2,3,100)
C = [0]
X, Y = np.meshgrid(xd, yd)
plt.figure()
plt.contour(X,Y,F1b([X,Y])[0],C,colors='red')
plt.contour(X,Y,F1b([X,Y])[1],C,colors='blue')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.grid(True)
plt.show()


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 4
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

''' INPUT DE LA FUNCIÓN "euler"

    f: función dependiente de t e y
    a,b: extremos del intervalo en que aproximamos la solución
    h: longitud de los subintervalos
    y0: valor inicial '''
    
def euler(f,a,b,h,y0):
    N = int((b-a)/h)
    if a+N*h < b:
        N += 1
        h = float(b-a)/N
        print('Se ha recalculado el valor de h =',h)
    t = np.linspace(a,b,N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range(N):
        y[i+1] = y[i] + h*f(t[i],y[i])
    return [t,y]


#  (a) 

def f4a(t,y):
    return (y/t)-(y/t)**2

h = 0.2
[t4a,y4a] = euler(f4a,1,2,h,1)


#  (b)

def f4b(t,y):
    return (2-2*t*y)/(1+t**2)

h = 0.2
[t4b,y4b] = euler(f4b,0,1,h,1)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 5
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#   (a)

def sol4a(t):
    return t/(1+np.log(t))

print('\nError verdadero: ',max(abs(y4a-sol4a(t4a))))

td = np.linspace(1,2,100)
plt.figure()
plt.title('Ejercicio 5a')
plt.plot(td,sol4a(td),label='Exact')
plt.plot(t4a,y4a,'or',label='Euler')
plt.plot(t4a,y4a,'r',label='Euler (approx)',alpha=0.25)
plt.legend()
plt.show()

#   (b)

def sol4b(t):
    return (2*t+1)/(t**2+1)

print('\nError verdadero: ',max(abs(y4b-sol4b(t4b))))

td = np.linspace(0,1,100)
plt.figure()
plt.title('Ejercicio 5b')
plt.plot(td,sol4b(td),label='Exact')
plt.plot(t4b,y4b,'or',label='Euler (nodos)')
plt.plot(t4b,y4b,'r',label='Euler (approx)',alpha=0.25)
plt.legend()
plt.show()

        
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 6
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

def RK4(f,a,b,h,y0):
    N = int((b-a)/h)
    if a+N*h < b:
        N += 1
        h = float(b-a)/N
        print('Se ha recalculado el valor de h =',h)
    t = np.linspace(a,b,N+1)
    y = np.zeros(N+1)
    y[0] = y0
    for i in range(N):
        s1 = h*f(t[i],y[i])
        s2 = h*f(t[i]+h/2,y[i]+s1/2)
        s3 = h*f(t[i]+h/2,y[i]+s2/2)
        s4 = h*f(t[i]+h,y[i]+s3)
        y[i+1] = y[i] + (s1+2*s2+2*s3+s4)/6
    return [t,y]


#  (a) 

def f4a(t,y):
    return (y/t)-(y/t)**2

def sol4a(t):
    return t/(1+np.log(t))

h = 0.2
[t4aE,y4aE] = euler(f4a,1,2,h,1)
[t4aR,y4aR] = RK4(f4a,1,2,h,1)
print('\nError verdadero con Euler: ',max(abs(y4aE-sol4a(t4aE))))
print('\nError verdadero con RK4: ',max(abs(y4aR-sol4a(t4aR))))

td = np.linspace(1,2,100)
plt.figure()
plt.title('Ejercicio 6a')
plt.plot(td,sol4a(td),label='Exact')
plt.plot(t4aE,y4aE,'or',label='Euler')
plt.plot(t4aE,y4aE,'r',alpha=0.25)
plt.plot(t4aR,y4aR,'og',label='Runge-Kutta-4')
plt.plot(t4aR,y4aR,'g',alpha=0.25)
plt.legend()
plt.show()


#  (b)

def f4b(t,y):
    return (2-2*t*y)/(1+t**2)

def sol4b(t):
    return (2*t+1)/(t**2+1)

h = 0.2
[t4bE,y4bE] = euler(f4b,0,1,h,1)
[t4bR,y4bR] = RK4(f4b,0,1,h,1)
print('\nError verdadero con Euler: ',max(abs(y4bE-sol4b(t4bE))))
print('\nError verdadero con RK4: ',max(abs(y4bR-sol4b(t4bR))))

td = np.linspace(0,1,100)
plt.figure()
plt.title('Ejercicio 6b')
plt.plot(td,sol4b(td),label='Exact')
plt.plot(t4bE,y4bE,'or',label='Euler')
plt.plot(t4bE,y4bE,'r',alpha=0.25)
plt.plot(t4bR,y4bR,'og',label='Runge-Kutta-4')
plt.plot(t4bR,y4bR,'g',alpha=0.25)
plt.legend()
plt.show()








