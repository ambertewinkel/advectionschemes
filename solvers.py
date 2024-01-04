# Module file with various solvers to use for the implicit schemes in schemes.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np

def Jacobi(A, xi, b, ni):
    """
    This function computes the solution x of the Ax=b linear matrix equation
    after n iterations with the Jacobi iteration method, which starts from 
    the initial guess xi. All arrays are assumed to be numpy.
    --- Input ---
    A   : matrix, NxN matrix where N is the length of both x and b vectors.
    xi  : initial guess for the x solution
    b   : 1D vector of length N, right-hand side of the equation.
    ni  : number of iterations to take
    --- Output --- 
    x   : 1D vector of length N.
    """
    N = len(b)
    x = np.zeros(N)
    x_old = xi.copy()
    
    # Check if dimensions are correct
    if A.shape[0] != N or A.shape[1] != N:
        raise IndexError(f'Matrix shape does not match vectors: shape = {A.shape}')
    if len(xi) != N:
        raise IndexError(f'Input vector shapes do not match each other: len(xi)!=len(b)')

    for it in range(ni):
        for i in range(N):
            ax = 0.0
            for j in range(N):
                if j != i:
                    ax += A[i,j]*x_old[j]
            x[i] = (b[i] - ax)/A[i,i]
        x_old = x.copy()

    return x

def GaussSeidel(A, xi, b, ni):
    """
    This function computes the solution x of the Ax=b linear matrix equation
    after n iterations with the Gauss-Seidel iteration method, which starts from 
    the initial guess xi. All arrays are assumed to be numpy.
    --- Input ---
    A   : matrix, NxN matrix where N is the length of both x and b vectors.
    xi  : initial guess for the x solution
    b   : 1D vector of length N, right-hand side of the equation.
    ni  : number of iterations to take
    --- Output --- 
    x   : 1D vector of length N.
    """
    N = len(b)
    x = xi.copy()
    
    # Check if dimensions are correct
    if A.shape[0] != N or A.shape[1] != N:
        raise IndexError(f'Matrix shape does not match vectors: shape = {A.shape}')
    if len(xi) != N:
        raise IndexError(f'Input vector shapes do not match each other: len(xi)!=len(b)')

    for it in range(ni):
        for i in range(N):
            ax = 0.0
            for j in range(N):
                if j != i:
                    ax += A[i,j]*x[j]
            x[i] = (b[i] - ax)/A[i,i]

    return x

def BackwardGaussSeidel(A, xi, b, ni):
    """
    This function computes the solution x of the Ax=b linear matrix equation
    after n iterations with the backward Gauss-Seidel iteration method, which starts from 
    the initial guess xi. All arrays are assumed to be numpy.
    --- Input ---
    A   : matrix, NxN matrix where N is the length of both x and b vectors.
    xi  : initial guess for the x solution
    b   : 1D vector of length N, right-hand side of the equation.
    ni  : number of iterations to take
    --- Output --- 
    x   : 1D vector of length N.
    """
    N = len(b)
    x = xi.copy()
    
    # Check if dimensions are correct
    if A.shape[0] != N or A.shape[1] != N:
        raise IndexError(f'Matrix shape does not match vectors: shape = {A.shape}')
    if len(xi) != N:
        raise IndexError(f'Input vector shapes do not match each other: len(xi)!=len(b)')

    for it in range(ni):
        for i in reversed(range(N)):
            ax = 0.0
            for j in reversed(range(N)):
                if j != i:
                    ax += A[i,j]*x[j]
            x[i] = (b[i] - ax)/A[i,i]

    return x

def SymmetricGaussSeidel(A, xi, b, ni):
    """
    This function computes the solution x of the Ax=b linear matrix equation
    after n iterations with the symmetric Gauss-Seidel iteration method, which starts from 
    the initial guess xi. All arrays are assumed to be numpy.
    --- Input ---
    A   : matrix, NxN matrix where N is the length of both x and b vectors.
    xi  : initial guess for the x solution
    b   : 1D vector of length N, right-hand side of the equation.
    ni  : number of iterations to take
    --- Output --- 
    x   : 1D vector of length N.
    """
    N = len(b)
    x = xi.copy()
    
    # Check if dimensions are correct
    if A.shape[0] != N or A.shape[1] != N:
        raise IndexError(f'Matrix shape does not match vectors: shape = {A.shape}')
    if len(xi) != N:
        raise IndexError(f'Input vector shapes do not match each other: len(xi)!=len(b)')

    for it in range(ni):
        # Forward Gauss-Seidel step
        for i in range(N):
            ax = 0.0
            for j in range(N):
                if j != i:
                    ax += A[i,j]*x[j]
            x[i] = (b[i] - ax)/A[i,i] # x is really xhalf

        # Backward Gauss-Seidel step
        for i in reversed(range(N)):
            ax = 0.0
            for j in reversed(range(N)):
                if j != i:
                    ax += A[i,j]*x[j] # x is really xhalf
            x[i] = (b[i] - ax)/A[i,i]

    return x