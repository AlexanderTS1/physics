# -*- coding: utf-8 -*-
"""
PRÁCTICA 9 - 03/04/2020

@authors: Julio Mulero  &  José Vicente  
"""

import numpy as np
import scipy.linalg as la


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 1
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

# INPUTS: A es una array de orden n (o nxm), i,j están en {0,1,2,...,n-1}
#         y c es un escalar. 

# Fij -> Intercambio de las filas i+1 y j+1.

def cambio_filas(A,i,j):
    subs = np.copy(A[i])
    A[i] = A[j]
    A[j] = subs
    return A

# Fi(c) -> Se multiplica la fila i+1 por el escalar c.
    
def prod_fila(A,i,c):
    subs = np.copy(A[i])
    A[i] = c*subs
    return A

# Fij(c) -> A la fila i+1 se le suma la fila j+1 multiplicada por el escalar c.
    
def suma_filas(A,i,j,c):
    subs = np.copy(A[i])
    A[i] = subs + c*A[j]
    return A

#   EJEMPLOS
    
A = np.identity(3)
len(A)
cambio_filas(A,0,2)
prod_fila(A,1,1./2)
suma_filas(A,2,0,3./2)

A = np.array([[1,0,2],[-1,2,1],[3,2,-2]],dtype=float)
len(A)
cambio_filas(A,0,2)
prod_fila(A,1,1/2)
suma_filas(A,2,0,3./2)

b = np.array([1.,0,2])
len(b)
cambio_filas(b,1,2)
prod_fila(b,0,1./2)
suma_filas(b,0,1,3./2)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 2
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# Solución para sistemas lineales cuadrados de la forma Ax = b cuya 
# matriz del sistema A es triangular superior.

def solucionU(A,b):
    n = len(A)
    x = np.zeros(n)
    for i in range(n-1,-1,-1):
        x[i] = (1/A[i,i])*(b[i]-np.sum(A[i]*x))
    return x


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 3
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# (1) Con el método de Gauss [con pivoteo parcial]
   
# Mediante las 3 transformaciones elementales que hemos visto arriba,
# obtenemos un sistema equivalente cuya matriz sea triangular superior
# y lo resolvemos usando la función 'solucionU' 
  
def gauss_parcial(A,b):
    n = len(A) 
    for i in range(n):
        k = np.argmax(abs(A[i::,i])) + i
        cambio_filas(b,i,k) 
        cambio_filas(A,i,k)
        ''' Si además quisiéramos tener unos en la diagonal principal, 
        entonces hemos de activar/quitar # en las siguientes lineas  '''
        # prod_fila(b,i,1/A[i][i])
        # prod_fila(A,i,1/A[i][i])
        for j in range(i+1,n):
            suma_filas(b,j,i,-A[j][i]/A[i][i]) 
            suma_filas(A,j,i,-A[j][i]/A[i][i]) 
    return solucionU(A,b)


A = np.array([[1,2,-3],[3,1,-2],[2,-3,1]],dtype=float)
b = np.array([-16,-10,-4.]) 
x2 = gauss_parcial(A,b)
print('La solución del sistema es: ',x2)

A
b
la.solve(A,b)

# Observamos cómo han cambiado las matrices A y b tras aplicar el método de Gauss.
# Vemos que, en efecto, se obtiene un sistema equivalente al original, 
# con matriz triangular superior.


# (2) Con la función 'solve' de la librería 'scipy.linalg'

A = np.array([[1,2,-3],[3,1,-2],[2,-3,1]],dtype=float)
b = np.array([-16,-10,-4.]) 
la.solve(A,b)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 4
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


# ¡Si no ponemos ninguna coma flotante puede dar ERROR, trabajamos con floats!
# O bien ponemos alguna coma flotante, o especificamos "dtype=float". 

A = np.array([[0,1,2],[1,1,-1],[2,1,0]],dtype=float)
Acopy = np.copy(A)
P = np.identity(len(A))

cambio_filas(A,0,1)
cambio_filas(P,0,1)

suma_filas(A,2,0,-2)
suma_filas(P,2,0,-2)

suma_filas(A,2,1,1)
suma_filas(P,2,1,1)

suma_filas(A,0,1,-1)
suma_filas(P,0,1,-1)

prod_fila(A,2,1/4.)
prod_fila(P,2,1/4.)

suma_filas(A,1,2,-2)
suma_filas(P,1,2,-2)

suma_filas(A,0,2,3)
suma_filas(P,0,2,3)

print("La inversa de la matriz\n",Acopy, "\n es:\n",P)

# Comprobación:

np.dot(Acopy,P)
np.dot(P,Acopy)


# Con la función 'inv' de la librería 'scipy.linalg'

A = np.array([[0,1,2],[1,1,-1],[2,1,0]],dtype=float)

la.inv(A)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 5
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


A=np.array([[7.,3,-1,2],[3,8,1,-4],[-1,1,4,-1],[2,-4,-1,6]])

# P es una matriz de permutaciones (eventualmente necesaria)
# L es una matriz triangular inferior con unos en la diagonal principal
# U es una matriz triangular superior

P, L, U = la.lu(A)

# Verificación:

np.dot(L,U)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 6
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

 
def jacobi(A,b,x0,norma,error,k):
    D = np.diag(np.diag(A))              # diagonal
    L = -np.tril(A-D)                    # triangular inferior
    U = -np.triu(A-D)                    # triangular superior
    M = D
    N = L+U
    B = np.dot(la.inv(M),N)              # M^{-1}N
    c = np.dot(la.inv(M),b)              # M^{-1}b
    val = la.eig(B)[0]                   # Autovalores de B
    ro = max(abs(val)) 
    if ro >= 1: 
        print("El método no es convergente")
        return[x0,0,ro]
    i=1                     # Si es convergente, comenzamos la aproximación.
    while True:
        if i>=k:
            print("El método no converge en",k,"pasos")
            return[x0,k,ro]
        x1 = np.dot(B,x0)+c            # Cada paso: x(m+1) = Bx(m) + c
        if la.norm(x1-x0,norma)<error: # Si el error es menor, salimos del bucle
            return[x1,i,ro]
        i = i+1                        # Si el error no es menor, continuamos
        x0 = x1.copy()                 # Guardamos en x0 el nuevo valor calculado


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 7
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
        
#   (a) 
        
A7a = np.array([[1,2,-3],[3,1,-2],[2,-3,1]],dtype=float)
b7a = np.array([-16,-10,-4],dtype=float)     
x0 = np.ones(3)

[x7a,i7a,ro7a] = jacobi(A7a,b7a,x0,np.inf,0.0001,100) 

# El radio espectral vale 3.4033 y, por tanto, el método NO es convergente.  

# Con la función 'solve' de la librería 'scipy.linalg'
   
la.solve(A7a,b7a)

#   (b) 
        
A7b = np.array([[5,1,2],[1,4,1],[2,2,5]],dtype=float)
b7b = np.array([1,2,3],dtype=float) 
x0 = np.zeros(3) 

[x7b,i7b,ro7b] = jacobi(A7b,b7b,x0,np.inf,0.00001,100)  

print('SOLUCIÓN:',x7b)
print('Número de Iteraciones:',i7b)
print('Radio espectral:',ro7b)
 
# Con la función 'solve' de la librería 'scipy.linalg'
   
la.solve(A7b,b7b)


# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
#   EJERCICIO 8
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


A8 = 12*np.eye(50) + np.diagflat(-2*np.ones((1,49)),k=-1) + np.diagflat(-2*np.ones((1,49)),k=1) + np.diagflat(np.ones((1,48)),k=-2) + np.diagflat(np.ones((1,48)),k=2)
b8 = 5*np.ones(50)

#  (a) Usando 'solve'

la.solve(A8,b8)

#  (b) Con el Método de Gauss

A8 = 12*np.eye(50) + np.diagflat(-2*np.ones((1,49)),k=-1) + np.diagflat(-2*np.ones((1,49)),k=1) + np.diagflat(np.ones((1,48)),k=-2) + np.diagflat(np.ones((1,48)),k=2)
b8 = 5*np.ones(50)

gauss_parcial(A8,b8)

#  (c) Con el Método de Jacobi

A8 = 12*np.eye(50) + np.diagflat(-2*np.ones((1,49)),k=-1) + np.diagflat(-2*np.ones((1,49)),k=1) + np.diagflat(np.ones((1,48)),k=-2) + np.diagflat(np.ones((1,48)),k=2)
b8 = 5*np.ones(50)
x0 = np.zeros(50) 

[x8J,i8J,ro8J] = jacobi(A8,b8,x0,np.inf,0.00001,100)

print('SOLUCIÓN:',x8J)
print('Número de Iteraciones:',i8J)
print('Radio espectral:',ro8J)


 