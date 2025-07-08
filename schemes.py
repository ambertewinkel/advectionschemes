# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, CTCS, Upwind, MPDATA, CNBS, and CNCS, and then BTBS, BTFS, BTCS with direct and iterative solvers. Lastly, hybrid scheme with BTBS + 1 Jacobi iteration for implicit and MPDATA for explicit
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


import numpy as np
import solvers as sv
import limiter as lim
from numba_config import jitflags
from numba import njit, prange
import matplotlib.pyplot as plt
import spatialdiscretisations as sd
import analytic as an
import logging


def FTBS(init, nt, dt, uf, dxc):
    """
    This function computes the FTBS (forward in time, backward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Setup and initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Time stepping
    for it in range(nt):
        field[it+1] = field[it] - dt*(np.roll(uf,-1)*field[it] - uf*np.roll(field[it],1))/dxc
 
    return field


def FTFS(init, nt, dt, uf, dxc):
    """
    This function computes the FTFS (forward in time, forward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Setup and initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Time stepping
    for it in range(nt):
        field[it+1] = field[it] - dt*(np.roll(uf*field[it],-1) - uf*field[it])/dxc

    return field


def CTCS(init, nt, dt, uf, dxc):
    """
    This function computes the CTCS (centered in time, centered in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect, defined at centers
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init Defined at centers
    """

    # Setup and initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # First time step is forward in time, centered in space (FTCS)
    field[1] = field[0] - 0.5*dt*(np.roll(uf,-1)*(field[0] - np.roll(field[0],-1)) - uf*(np.roll(field[0],1) + field[0]))/dxc

    # Time stepping
    for it in range(1, nt):
        field[it+1] = field[it-1] - dt*(np.roll(uf,-1)*(field[it] - np.roll(field[it],-1)) - uf*(np.roll(field[it],1) + field[it]))/dxc
        
    return field


@njit(**jitflags)
def Upwind(init, nt, dt, uf, dxc): # FTBS when u >= 0, FTFS when u < 0
    """
    This function computes the upwind (FTBS when u>=0, FTFS when u<0)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed. dt and dx are assumed to be positive.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Setup and initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    
    cc = 0.5*dt*(uf + np.roll(uf,-1))/dxc # sum for cc[i] is over faces i-1/2 and i+1/2

    # Time stepping
    for it in range(nt):
        spatial = np.where(cc >= 0., np.roll(uf,-1)*field[it] - uf*np.roll(field[it],1), np.roll(uf*field[it],-1) - uf*field[it]) # BS when u >= 0, FS when u < 0
        field[it+1] = field[it] - dt*spatial/dxc
    
    return field


@njit(**jitflags)
def BTBS(init, nt, dt, uf, dxc):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = np.linalg.solve(M, field[it])

    return field


def BTBS_Jacobi(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.Jacobi(M, field[it], field[it], niter)

    return field


def BTBS_GaussSeidel(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.GaussSeidel(M, field[it], field[it], niter)

    return field


def BTBS_SymmetricGaussSeidel(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the symmetric Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, desfault=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.SymmetricGaussSeidel(M, field[it], field[it], niter)

    return field


def BTFS(init, nt, dt, uf, dxc):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = np.linalg.solve(M, field[it])
    
    return field


def BTFS_Jacobi(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.Jacobi(M, field[it], field[it], niter)

    return field


def BTFS_GaussSeidel(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.BackwardGaussSeidel(M, field[it], field[it], niter)

    return field


def BTFS_SymmetricGaussSeidel(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the symmetric Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.SymmetricGaussSeidel(M, field[it], field[it], niter)

    return field


def BTCS(init, nt, dt, uf, dxc):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = np.linalg.solve(M, field[it])

    return field


def BTCS_Jacobi(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.Jacobi(M, field[it], field[it], niter)

    return field


def BTCS_GaussSeidel(init, nt, dt, uf, dxc, niter=1):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field[it+1] = sv.GaussSeidel(M, field[it], field[it], niter)

    return field


def CNBS(init, nt, dt, uf, dxc):
    """
    This functions implements the CNBS scheme (Crank-Nicolson in i.e. trapezoidal implicit, backward in 
    space), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + 0.5*dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -0.5*dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        rhs = (1 - 0.5*dt*np.roll(uf,-1)/dxc)*field[it] + 0.5*dt*uf*np.roll(field[it],1)/dxc
        field[it+1] = np.linalg.solve(M, rhs)

    return field


def CNCS(init, nt, dt, uf, dxc):
    """
    This functions implements the CNCS scheme (Crank-Nicolson in i.e. trapezoidal implicit, centered in 
    space), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + 0.25*dt*(np.roll(uf,-1)[i] - uf[i])/dxc[i]
        M[i, i-1] = -0.25*dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = 0.25*dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        rhs = (1 - 0.25*dt*(np.roll(uf,-1) - uf)/dxc)*field[it] + 0.25*dt*uf*np.roll(field[it],1)/dxc - 0.25*dt*np.roll(uf,-1)*np.roll(field[it],-1)/dxc
        field[it+1] = np.linalg.solve(M, rhs)

    return field


@njit(**jitflags)
def MPDATA(init, nt, dt, uf, dxc, eps=1e-16, do_limit=False, limit=0.5, nSmooth=0):
    """
    This functions implements the MPDATA scheme without a gauge, assuming a 
    constant velocity (input through the Courant number) and a 
    periodic spatial domain.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_limit: boolean, optional. If True, V is limited. Default is True.
    limit   : float, optional. Limiting value. Default is 0.5.
    nSmooth : integer, optional. Number of smoothing steps for V. Default is 1.
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    dxf = 0.5*(dxc + np.roll(dxc,1))
    dx_up = np.zeros(len(init))
    flx_FP, flx_SP = np.zeros(len(init)), np.zeros(len(init))

    # Time stepping
    for it in prange(nt):
        # First pass  
        for i in range(len(init)):
            flx_FP[i] = flux_njit(np.roll(field[it],1)[i], field[it,i], uf[i]) # flx_FP[i] is at i-1/2
        field_FP = field[it] - dt*(np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        for i in range(len(init)):
            dx_up[i] = 0.5*flux_njit(np.roll(dxc,1)[i], dxc[i], uf[i]/abs(uf[i]))
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps) # A[i] is at i-1/2
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf) # Same index shift as for A
        
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))
        
        for i in range(len(init)):
            flx_SP[i] = flux_njit(np.roll(field_FP,1)[i], field_FP[i], V[i])
        field[it+1] = field_FP - dt*(np.roll(flx_SP,-1) - flx_SP)/dxc

    return field


@njit(**jitflags)
def MPDATA_gauge_njit(init, nt, dt, uf, dxc, corrsource='firstpass', FCT=False):
    """
    This functions implements the MPDATA scheme with an infinite gauge, assuming a 
    constant velocity (input through the Courant number) and a 
    periodic spatial domain.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    dxf = 0.5*(dxc + np.roll(dxc,1))
    flx_FP, flx_SP = np.zeros(len(init)), np.zeros(len(init))
    dx_up = np.zeros(len(init))

    # Time stepping
    for it in range(nt):
        # First pass  
        for i in range(len(init)):
            flx_FP[i] = flux_njit(np.roll(field[it],1)[i], field[it,i], uf[i])
        field_FP = field[it] - dt*(np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        for i in range(len(init)):
            dx_up[i] = 0.5*flux_njit(np.roll(dxc,1)[i], dxc[i], uf[i]/abs(uf[i]))
        
        # Use the first-pass or the previous field for the correction 
        if corrsource == 'firstpass':
            V = 0.5*(field_FP - np.roll(field_FP,1))*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf)   # V[i] is at i-1/2
        elif corrsource == 'previous':
            V = 0.5*(field[it] - np.roll(field[it],1))*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf)   # V[i] is at i-1/2

        flx_SP = V
        #if FCT == True:
        #    lim.FCT(field_FP, flx_SP, dxc, previous=True, field=field)
        
        field[it+1] = field_FP - dt*(np.roll(flx_SP,-1) - flx_SP)/dxc

    return field


def MPDATA_gauge(init, nt, dt, uf, dxc, corrsource='firstpass', FCT=False, returndiffusive=False): # not numba yet to allow for FCT implementation which isn't numba yet
    """
    This functions implements the MPDATA scheme with an infinite gauge, assuming a 
    constant velocity (input through the Courant number) and a 
    periodic spatial domain.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    finalfield_FP = np.zeros(np.shape(field))
    finalfield_FP[0] = init.copy()

    dxf = 0.5*(dxc + np.roll(dxc,1))
    flx_FP = np.zeros(len(init))
    dx_up = np.zeros(len(init))

    # Time stepping
    for it in range(nt):
        # First pass  
        for i in range(len(init)):
            flx_FP[i] = dt*flux_njit(np.roll(field[it],1)[i], field[it,i], uf[i])
        field_FP = field[it] - (np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        for i in range(len(init)):
            dx_up[i] = 0.5*flux_njit(np.roll(dxc,1)[i], dxc[i], uf[i]/abs(uf[i]))
        
        # Use the first-pass or the previous field for the correction
        # V = flux for the second-pass flx_SP
        if corrsource == 'firstpass':
            V = 0.5*dt*(field_FP - np.roll(field_FP,1))*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf)   # V[i] is at i-1/2 
        elif corrsource == 'previous':
            V = 0.5*dt*(field[it] - np.roll(field[it],1))*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf)   # V[i] is at i-1/2

        if returndiffusive == True:
            finalfield_FP[it+1] = field_FP.copy()
        
        # Limiting the second-pass correction
        if FCT == True:
            V = lim.FCT(field_FP, V, dxc)
        
        field[it+1] = field_FP - (np.roll(V,-1) - V)/dxc

    if returndiffusive == True:
        return finalfield_FP
    else:
        return field


def aiMPDATA1J(init, nt, dt, uf, dxc, do_beta='switch', eps=1e-16, do_limit=False, limit=0.5, nSmooth=0, gauge=0.):
    """
    This functions implements 
    Explicit: MPDATA scheme (without a gauge, assuming a 
    constant velocity and a 
    periodic spatial domain)
    Implicit: Upwind with 1 Jacobi iteration
    Optional limit and smoothing for V for the explicit second pass of MPDATA.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_beta : string, optional. If 'switch', beta is 0 for explicit and 1 for implicit. If 'blend', beta is a blend between 0 and 1. Default is 'switch'.
    do_limit: boolean, optional. If True, V is limited. Default is True.
    limit   : float, optional. Limiting value. Default is 0.5.
    nSmooth : integer, optional. Number of smoothing steps for V. Default is 0.
    gauge   : float, optional. Gauge term. Default is 0.
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        # beta[i] is at i-1/2 # 0: explicit, 1: implicit 
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1.))
    elif do_beta == 'blend':
        # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)])
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(cc))) # chi[i] is at i-1/2
    
    # Time stepping
    for it in range(nt):
        # First pass
        # flx_FP[i] is at i-1/2 # upwind
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # Determine whether implicit or explicit (or blend)
        for i in range(len(cc)):
            if beta[i] != 0. or np.roll(beta,-1)[i] != 0.: # Upwind implicit with 1 Jacobi iteration
                aii = 1. + (np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])*dt/dxc[i]
                aiim1 = -dt*beta[i]*ufp[i]/dxc[i]
                aiip1 = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
                field_FP[i] = (rhs[i] - aiim1*np.roll(field[it],1)[i] - aiip1*np.roll(field[it],-1)[i])/aii           
            else:
                field_FP[i] = rhs[i]

        field_FP = field_FP + gauge

        # Second pass
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # A[i] is at i-1/2
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps)
        # Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)
        
        # Limit V
        if do_limit == True:
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)

        # Smoothing
        for iSmooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc - gauge      
                
    return field


def imMPDATA(init, nt, dt, uf, dxc, eps=1e-16, solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0, gauge=0.):
    """
    Implements MPDATA with an implicit first pass. 
    First pass: implicit upwind with numpy direct elimination on the whole matrix.
    Second pass: explicit MPDATA correction. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    Optional: smooth and limit V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_limit: boolean, optional. If True, limit the antidiffusive velocity V
    limit   : float, optional. Assumed positive. Limiting value for V
    nSmooth : integer, optional. Number of smoothing iterations for V
    gauge   : float, optional. Gauge term to add to the field
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    solverfn = getattr(sv, solver)

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))
    
    # For stability, determine the amount of temporal MPDATA correction in V for use later
    beta = np.ones(len(init)) # beta[i] is at i-1/2; implicit
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # First pass: converged BTBS (implicit)
        field_FP = solverfn(M, field[it], rhs, niter) + gauge

        # Second pass (explicit)
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # Use the first-pass field for A. A[i] is at i-1/2
        A = (field_FP - np.roll(field_FP,1))\
            /(field_FP + np.roll(field_FP,1) + eps)
        
        # Calculate and limit the antidiffusive velocity V. Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)        
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        # Calculate the flux and second-pass result
        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc - gauge

    return field


def imMPDATA_gauge(init, nt, dt, uf, dxc, eps=1e-16, solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0):
    """
    Implements infinite-gauge MPDATA with an implicit first pass. 
    First pass: implicit upwind with numpy direct elimination on the whole matrix.
    Second pass: explicit MPDATA correction with an infinite-gauge. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    Optional: smooth and limit V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_limit: boolean, optional. If True, limit the antidiffusive velocity V
    limit   : float, optional. Assumed positive. Limiting value for V
    nSmooth : integer, optional. Number of smoothing iterations for V
    gauge   : float, optional. Gauge term to add to the field
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    solverfn = getattr(sv, solver)

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))
    
    # For stability, determine the amount of temporal MPDATA correction in V for use later
    beta = np.ones(len(init)) # beta[i] is at i-1/2; implicit
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # First pass: converged BTBS (implicit)
        field_FP = solverfn(M, field[it], rhs, niter)

        # Second pass (explicit)
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # Use the first-pass field for A. A[i] is at i-1/2
        A = (field_FP - np.roll(field_FP,1))/2.
        
        # Calculate and limit the antidiffusive velocity V. Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)        
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        # Calculate the flux and second-pass result
        flx_SP = flux(1., 1., V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc

    return field


def aiMPDATA(init, nt, dt, uf, dxc, eps=1e-16, do_beta='switch', solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0, gauge=0.):
    """
    Implements a hybrid scheme with explicit MPDATA correction.  
    First pass: explicit or implicit (or both if do_beta='blend') upwind with numpy direct elimination on the whole matrix. beta determines the degree of im/ex - as trapezoidal implicit.
    Second pass: explicit MPDATA correction. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    V is limited to +-corrClimit. Another difference is the option (i.e., commented out) of smoothing V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_beta : string, optional. If 'switch', beta is 0 for explicit and 1 for implicit. If 'blend', beta is a blend between 0 and 1.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_limit: boolean, optional. If True, limit the antidiffusive velocity V
    limit   : float, optional. Assumed positive. Limiting value for V
    nSmooth : integer, optional. Number of smoothing iterations for V
    gauge   : float, optional. Gauge term to add to the field
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    solverfn = getattr(sv, solver)

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # For stability, determine the amount of temporal MPDATA correction in V for use later
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # First pass: converged BTBS - dependent on the Courant number and beta.
        field_FP = solverfn(M, field[it], rhs, niter) + gauge

        # Second pass
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # Use the first-pass field for A. A[i] is at i-1/2
        A = (field_FP - np.roll(field_FP,1))\
            /(field_FP + np.roll(field_FP,1) + eps)
        
        # Calculate and limit the antidiffusive velocity V. Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)

        if do_limit == True: # Limit V
            corrCLimit = np.absolute(limit*uf)
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        # Calculate the flux and second-pass result
        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc - gauge

    return field


def aiMPDATA_gauge(init, nt, dt, uf, dxc, do_beta='switch', solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0, third_order=False, corrsource='previous'):
    """
    Implements a hybrid scheme with explicit infinite-gauge MPDATA correction.  
    First pass: explicit or implicit (or both if do_beta='blend') upwind with numpy direct elimination on the whole matrix. beta determines the degree of im/ex - as trapezoidal implicit.
    Second pass: explicit MPDATA correction with an infinite-gauge. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    V is limited to +-corrClimit. Another difference is the option (i.e., commented out) of smoothing V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_beta : string, optional. If 'switch', beta is 0 for explicit and 1 for implicit. If 'blend', beta is a blend between 0 and 1.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_limit: boolean, optional. If True, limit the antidiffusive velocity V
    limit   : float, optional. Assumed positive. Limiting value for V
    nSmooth : integer, optional. Number of smoothing iterations for V
    gauge   : float, optional. Gauge term to add to the field
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init    
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    solverfn = getattr(sv, solver)

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # For stability, determine the amount of temporal MPDATA correction in V for use later
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # First pass: converged BTBS - dependent on the Courant number and beta.
        field_FP = solverfn(M, field[it], rhs, niter)

        # Second pass
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # Use the first-pass or the previous field for A. A[i] is at i-1/2
        if corrsource == 'firstpass':
            A = (field_FP - np.roll(field_FP,1))/2.
        elif corrsource == 'previous':
            A = (field[it] - np.roll(field[it],1))/2.
        
        # Calculate and limit the antidiffusive velocity V. Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        # Calculate the flux and second-pass result
        flx_SP = flux(1., 1., V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc
        if third_order == True:
            field[it+1] += thirdorder(field[it], uf, dxc, dt) # check where the minus/plus comes from

    return field


def aiMPDATA_gauge_solverlast(init, nt, dt, uf, dxc, do_beta='switch', solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0, third_order=False): # !!! 24072024, check for stability as this matches the stability derivation
    """
    Implements a hybrid scheme with explicit infinite-gauge MPDATA correction.  
    First pass: explicit or implicit (or both if do_beta='blend') upwind with numpy direct elimination on the whole matrix. beta determines the degree of im/ex - as trapezoidal implicit.
    Second pass: explicit MPDATA correction with an infinite-gauge. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    V is limited to +-corrClimit. Another difference is the option (i.e., commented out) of smoothing V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_beta : string, optional. If 'switch', beta is 0 for explicit and 1 for implicit. If 'blend', beta is a blend between 0 and 1.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_limit: boolean, optional. If True, limit the antidiffusive velocity V
    limit   : float, optional. Assumed positive. Limiting value for V
    nSmooth : integer, optional. Number of smoothing iterations for V
    gauge   : float, optional. Gauge term to add to the field
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init    
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))

    solverfn = getattr(sv, solver)

    dxf = 0.5*(dxc + np.roll(dxc,1)) # dxf[i] is at i-1/2
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # For stability, determine the amount of temporal MPDATA correction in V for use later
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 # also tested on 18-12-2024 with just chi=1-2*beta

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc

        # Second pass
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        # Use the first-pass field for A. A[i] is at i-1/2
        A = (field[it] - np.roll(field[it],1))/2.
        
        # Calculate and limit the antidiffusive velocity V. Same index shift as for A
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*chi*uf)
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        # Calculate the flux and second-pass result
        flx_SP = flux(1., 1., V)
        rhs += dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc
        if third_order == True:
            toc = thirdorder(field[it], uf, dxc, dt)
            rhs += toc
        field[it+1] = solverfn(M, field[it], rhs, niter)

    return field


def aiMPDATA_gauge_clt1(init, nt, dt, u, dx, solver='NumPy', do_beta='', do_limit='', nSmooth=0, solver_location='first-pass', niter=1):
    """
    Implements a hybrid scheme with explicit infinite-gauge MPDATA correction.  
    First pass: explicit or implicit (or both if do_beta='blend') upwind with numpy direct elimination on the whole matrix. beta determines the degree of im/ex - as trapezoidal implicit.
    Second pass: explicit MPDATA correction with an infinite-gauge. For this:
    A is calculated with field_FP (first-pass) and not field[it]. The upwind direction is determined with the pseudovelocity V.
    V is limited to +-corrClimit. Another difference is the option (i.e., commented out) of smoothing V. 
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    u      : array of floats, velocity defined at faces
    dx     : array of floats, spacing between cell faces
    eps     : float, optional. Small number to avoid division by zero.
    do_beta : string, optional. If 'switch', beta is 0 for explicit and 1 for implicit. If 'blend', beta is a blend between 0 and 1.
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    niter   : integer, optional. Number of iterations for Jacobi solver.
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init    
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    field_FP = np.zeros(len(init))
    solverfn = getattr(sv, solver)
    
    # For stability, determine the amount of temporal MPDATA correction in V for use later
    c = 0.5*dt*(np.roll(u,-1) + u)/dx
    #beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)])
    chi = 2./c - 1.

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + (c[i]-1)
        M[i,(i-1)%len(init)] = -(c[i]-1)
    
    # Time stepping
    for it in range(nt):
        # First pass: upwind. flx_FP[i] is at i-1/2
        rhs = np.roll(field[it],1)
        if solver_location == 'first-pass':
            field_FP = solverfn(M, field[it], rhs, niter)

        # Second pass, calculate the antidiffusive velocity V. Same index shift as for A
        V = 0.5*u/(0.5*dx)*(0.5*dx - 0.5*dt*chi*u)*(field[it] - np.roll(field[it],1))
        # Calculate the flux and second-pass result
        if solver_location =='first-pass':
            field[it+1] = field_FP + dt*(-np.roll(V,-1) + V)/dx
        elif solver_location == 'second-pass':
            rhs += dt*(-np.roll(V,-1) + V)/dx
            field[it+1] = solver(M, field[it], rhs, niter)

    return field
    

def implicitLW(init, nt, dt, u, dx):
    """This function implements the implicit Lax-Wendroff scheme, which should have second-order accuracy in space and time.
    Assumes constant velocity and uniform grid. The numpy solver is employed at the end of each time step."""

    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    c = u*dt/dx
    
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)):
        M[i,i-1] = 0.5*(c[i]*c[i] - c[i])
        M[i,i] = 1 - c[i]*c[i]
        M[i,(i+1)%len(init)] = 0.5*(c[i]*c[i] + c[i])

    # Time stepping
    for it in range(nt):
        rhs = field[it]
        field[it+1] = sv.NumPy(M, field[it], rhs, ni=1)

    return field


def LW_aicorrection(init, nt, dt, u, dx, solver='NumPy', niter=0):
    """This scheme is like aiMPDATA_gauge but with a different correction. Namely, it has an implicit correction in addition to the explicit one in aiMPDATA_gauge."""


    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()
    
    solverfn = getattr(sv, solver)
    c = dt*u/dx
    theta = 1-1/c

    M = np.zeros((len(init), len(init)))

    D = np.zeros(len(init))
    D = 0.5*((1-2*theta)*c*c -c)
    for i in range(len(init)):
        M[i,i] = 1 + theta[i]*c[i] + 2*theta[i]*D[i]
        M[i,(i-1)%len(init)] = -theta[i]*c[i] - theta[i]*D[i]
        M[i,(i+1)%len(init)] = -theta[i]*D[i] 

    for it in range(nt):
        rhs = field[it] + c*(1 - theta)*(np.roll(field[it],1) - field[it]) + D*(1 - theta)*(np.roll(field[it],1) - 2*field[it] + np.roll(field[it],-1))
        field[it+1] = solverfn(M, field[it], rhs, niter)

    return field


def butcherExaiUpwind():
    # I.e., forward Euler Butcher tableau
    A = np.array([[0., 0.],[1., 0.]])
    b = np.array([[1.,0.]])
    return A, b


def butcherImaiUpwind():
    # I.e., backward Euler Butcher tableau
    A = np.array([[0., 0.],[0., 1.]])
    b = np.array([[0., 1.]])
    return A, b


@njit(**jitflags)
def aiUpwind(init, nt, dt, uf, dxc, solver='NumPy', niter=0):
    """This scheme test the accuracy of adaptively implicit upwind. (Needs to be first-order accurate to have a nice second/third-order correction to it.)
    Currently not upwind just FTBS - i.e. not accounting for the sign of u.
    Assuming constant dx."""
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    cf = uf*dt/dxc # [i] at i-1/2
    cc_out = 0.5*(np.abs(uf) - uf + np.abs(np.roll(uf,-1)) + np.roll(uf,-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *outward* pointing velocities
    cc_in = 0.5*(np.abs(uf) + uf + np.abs(np.roll(uf,-1)) - np.roll(uf,-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *inward* pointing velocities
    betac_out = np.maximum(0., 1.-1./cc_out)
    betac_in = np.maximum(0., 1.-1./cc_in)
    beta = np.maximum(np.maximum(betac_out, np.roll(betac_out,1)), np.maximum(betac_in, np.roll(betac_in, 1))) # [i] at i-1/2
    
    M = np.zeros((len(init), len(init)))
    for i in prange(len(init)):
        M[i,i] = 1. + beta[(i+1)%len(init)]*cf[(i+1)%len(init)]
        M[i,i-1] = -1.*beta[i]*cf[i] 
        #M[i,i-1] = -1.*beta[(i+1)%nx]*cf[i] # Using this makes the AdImEx boundary artefact disappear but also makes it nonconservative
    
    for it in prange(nt):
        rhs = field[it] - (np.roll(cf*(1-beta),-1)*field[it] - cf*(1-beta)*np.roll(field[it],1))
        #rhs = field[it] - (1-beta)*(cf*field[it] - np.roll(cf*field[it],1)) # Using this makes the AdImEx boundary artefact disappear but also makes it nonconservative
        field[it+1] = np.linalg.solve(M, rhs)
    
    return field


def LW3(init, nt, dt, uf, dxc, FCT=False, returndiffusive=False): # Only explicit and uniform grid and velocity # previously called thirdorderinfgaugeLWMPDATA
    """This scheme is based on MPDATA_LW3 derivation from Hilary from 29-07-2024. Third-order Lax-Wendroff.
    It assumes the grid is uniform.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output ---
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """    
    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    finalfield_FP = np.zeros(np.shape(field))
    finalfield_FP[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)

    for it in range(nt):
        flx_FP = uf*dt*np.roll(field[it],1) # assumes u>0, i.e., FTBS rather than upwind # flx_FP[i] defined at i-1/2
        field_FP = field[it] - (np.roll(flx_FP,-1) - flx_FP)/dxc
        
        flx_SP = 0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        field_SP = field_FP - (np.roll(flx_SP,-1) - flx_SP)/dxc

        flx_TP = - uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2
        field[it+1] = field_SP - (np.roll(flx_TP,-1) - flx_TP)/dxc

        if FCT == True:
            corr = flx_SP + flx_TP # high-order flux - low-order flux # corr[i] defined at i-1/2
            corr = lim.FCT(field_FP, corr, dxc, previous=field[it])
            field[it+1] = field_FP - (np.roll(corr,-1) - corr)/dxc
            
        if returndiffusive == True:
            finalfield_FP[it+1] = field_FP.copy()

    if returndiffusive == True:
        return finalfield_FP
    else:
        return field


def iLW3(init, nt, dt, uf, dxc, solver='NumPy', niter=1, FCT=False, returndiffusive=False):
    """This function implements the third-order Lax-Wendroff scheme with an implicit first-order part. The second- and third-order correction are based on the previous time step's field, and are not actually be second- and third-order in combination with first-order implicit (only with first-order (explicit) upwind).
    Assumes constant velocity and uniform grid.
    Third-order correction is based on MPDATA_LW3 derivation from Hilary from 29-07-2024. Third-order Lax-Wendroff.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output ---
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """    
    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    finalfield_FP = np.zeros(np.shape(field))
    finalfield_FP[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)
    
    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2; implicit
    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))
    previous = np.full(len(init), None) 

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: implicit upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf) # this is not actually the first-order flux here. That is the M matrix above. However, corr below and as such FCT is still correct.
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc
        field_FP = solverfn(M, field[it], rhs, niter)
        
        # Second pass
        flx_SP = 0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        field_SP = field_FP - (np.roll(flx_SP,-1) - flx_SP)/dxc

        # Third pass
        flx_TP = - uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2
        field[it+1] = field_SP - (np.roll(flx_TP,-1) - flx_TP)/dxc

        previous = [field[it][i] if c[i] <= 1. else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.

        if FCT == True:
            corr = flx_SP + flx_TP # high-order flux - low-order flux # corr[i] defined at i-1/2
            corr = lim.FCT(field_FP, corr, dxc, previous)
            field[it+1] = field_FP - (np.roll(corr,-1) - corr)/dxc
            
        if returndiffusive == True:
            finalfield_FP[it+1] = field_FP.copy()

    if returndiffusive == True:
        return finalfield_FP
    else:
        return field
    

def aiLW3(init, nt, dt, uf, dxc, do_beta='switch', solver='NumPy', niter=1, FCT=False, aiFCT=False, returndiffusive=False):
    """This function implements the third-order Lax-Wendroff scheme with an adaptively implicit first-order part. The second- and third-order correction are based on the previous time step's field, and are not actually be second- and third-order in combination with first-order implicit (only with first-order (explicit) upwind).
    Assumes constant velocity and uniform grid.
    Third-order correction is based on MPDATA_LW3 derivation from Hilary from 29-07-2024. Third-order Lax-Wendroff. 
    """    
    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    finalfield_FP = np.zeros(np.shape(field))
    finalfield_FP[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)
    
    if aiFCT == True: # use adaptively implicit FCT (i.e., with different max/min limits used in FCT)
        #fFCT = # somehow change the FCT arguments outside of the time for loop
        pass

    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2
    previous = np.full(len(init), None) 

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # First pass: implicit upwind. flx_FP[i] is at i-1/2
        flx_FP = flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc
        field_FP = solverfn(M, field[it], rhs, niter)
        
        # Second pass
        flx_SP = 0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        field_SP = field_FP - (np.roll(flx_SP,-1) - flx_SP)/dxc

        # Third pass
        flx_TP = - uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2
        field[it+1] = field_SP - (np.roll(flx_TP,-1) - flx_TP)/dxc
        
        previous = [field[it][i] if c[i] <= 1. else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.

        if FCT == True:
            corr = flx_SP + flx_TP # high-order flux - low-order flux # corr[i] defined at i-1/2
            corr = lim.FCT(field_FP, corr, dxc, previous)
            field[it+1] = field_FP - (np.roll(corr,-1) - corr)/dxc
            
        if returndiffusive == True:
            finalfield_FP[it+1] = field_FP.copy()

    if returndiffusive == True:
        return finalfield_FP
    else:
        return field
    

def HW_aiMPDATA(init, nt, dt, uf, dxc, do_beta='switch', do_limit=False, limit=0.5, nSmooth=0, gauge=0.):
    """Code from Hilary Weller (April 2024) for a hybrid MPDATA scheme. Based on implicitMPDATA.py.
    It assumes the grid is uniform."""
    nx = 40
    carr = dt*uf/dxc # assumes uniform grid
    c = carr[0].copy()

    # Initialise and advect profile phi based on space, x
    x = np.arange(0., 1., 1/nx)
    
    # Sets of parameter values to compare and output files for results
    params = \
    [{"beta":0, "gauge":gauge, "nCorr":1, "corrOption":"new", 'do_beta': do_beta, 'do_limit':do_limit, "limit":limit, 'nSmooth':nSmooth}]
        
    # Advection time steps using MPDATA and plotting every time step
    phi = np.zeros((nt+1, len(init)))
    phi[0] = init.copy()
    for it in range(nt):
        phi[it+1] = HW_aiMPDATA_timestepping(phi[it], c, **params[0])

    return phi


def HW_aiMPDATA_timestepping(phi, c, beta = 0., do_beta = 'switch', gauge = 0., nCorr = 1, eps = 1e-16,
          corrOption = "new", do_limit = False, limit = 0.5, nSmooth = 0):
    """
    Advects 1d profile phi for one time step with Courant number c using MPDATA
    with periodic boundary conditions and nCorr higher order corrections.
    If beta = 0 and c<1 then the scheme uses standard, fully explicit MPDATA.
    If beta>0 then the scheme is a correction on trapezoidal implicit with off
    centering beta. If gauge>0 then a finite gauge is used.
    This scheme assumes do_beta='blend'.
    --- Input ---
    phi    : 1d array of floats to be advected
    c      : float. Courant number = dt u/dx
    beta   : float between 0 and one. Implicit off centering
    gauge  : float
    nCorr  : int >= 0
    eps    : float. To prevent division by zero
    corrOption : string. "old" to use old value to calculate the anti-diffusive
                         flux and "new" to use the latest value.
    corrCLimit : float. Courant limit for the anti-diffusive flux
    nSmooth : int. Number of smoothing steps to apply to the anti-diffusive flux
    --- Output ---
    phiNew : modified 1d array phi after one time step
    """
    
    nx = len(phi)
    
    # Set beta as needed for implicitness
    if do_beta == 'switch':
        if (abs(c) > 1) & (beta == 0):
            beta = 1. 
    elif do_beta == 'blend':
        if (abs(c) > 1) & (beta == 0):
            beta = 1 - 1/abs(c) 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = max(1 - 2*beta, 0)

    # Set up the matrix for the trapezoidal implicit, first-order step
    M = np.zeros([nx, nx])
    for i in range(nx):
        M[i,i] = 1 + beta*abs(c)
        M[i,(i-1)%nx] = -beta*max(c,0)
        M[i,(i+1)%nx] = beta*min(c,0)
    
    # RHS of the matrix equation
    RHS = phi - (1-beta)*max(c,0)*(phi - np.roll(phi,1)) \
              - (1-beta)*min(c,0)*(np.roll(phi,-1) - phi)

    # First-order step
    phiNew = np.linalg.solve(M, RHS) + gauge

    # Values of phi to use for the anti-diffusive flux (no additional memory)
    phiF = phiNew if (corrOption == "new") else phi
    
    # MPDATA correction steps
    for icorr in range(nCorr):
        # Anti-diffusive velocities (label i is at position i-1/2)
        v = (c - chi*c**2)*(phiF - np.roll(phiF,1))\
                      /(phiF + np.roll(phiF,1) + eps)
                
        # Limit the anti-diffusive fluxes to obey the Courant limit
        if do_limit==True:
            v = np.maximum(np.minimum(v, limit), -limit)
        
        # Smoothing
        for ismooth in range(nSmooth):
            v = 0.5*v + 0.25*(np.roll(v,1) + np.roll(v,-1))

        # Correction
        phiNew -= np.maximum(np.roll(v,-1),0.)*phiNew \
                + np.minimum(np.roll(v,-1),0.)*np.roll(phiNew,-1) \
                - np.maximum(v,0)*np.roll(phiNew,1) \
                - np.minimum(v,0)*phiNew
        
        # For any subsequent corrections, phiNew is to be used for v
        phiF = phiNew
    
    return phiNew - gauge


def hybrid_Upwind_BTBS1J(init, nt, dt, uf, dxc, do_beta='switch'):
    """
    This functions implements 
    Explicit: upwind scheme (assuming a 
    constant velocity and a 
    periodic spatial domain)
    Implicit: BTBS with 1 Jacobi iteration
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit 
    elif do_beta == 'blend':
        # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)])
    else:
        print('Error: do_beta must be either "switch" or "blend"')

    # Time stepping
    for it in range(nt):
        flx = flux(np.roll(field[it],1), field[it], uf) # flx[i] is at i-1/2 # upwind
        rhs = field[it] - dt*(np.roll((1. - beta)*flx,-1) - (1. - beta)*flx)/dxc
        # for ... # include number of iterations here!!!
        for i in range(len(cc)):
            if beta[i] != 0. or np.roll(beta,-1)[i] != 0.: # BTBS1J
                aii = 1 + np.roll(beta*uf,-1)[i]*dt/dxc[i]
                aiim1 = -dt*beta[i]*uf[i]/dxc[i]
                field[it+1,i] = (rhs[i] - aiim1*np.roll(field[it],1)[i])/aii
            else:
                field[it+1,i] = rhs[i]

    return field


def hybrid_Upwind_Upwind1J(init, nt, dt, uf, dxc, do_beta='switch'):
    """
    This functions implements 
    Explicit: upwind scheme (assuming a constant velocity and a 
    periodic spatial domain)
    Implicit: Upwind with 1 Jacobi iteration
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """
    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit 
    elif do_beta == 'blend':
        # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)])
    else:
        print('Error: do_beta must be either "switch" or "blend"')

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Time stepping
    for it in range(nt):
        flx = flux(np.roll(field[it],1), field[it], uf) # flx[i] is at i-1/2
        rhs = field[it] - dt*(np.roll((1. - beta)*flx,-1) - (1. - beta)*flx)/dxc
        for i in range(len(cc)):
            if beta[i] != 0. or np.roll(beta,-1)[i] != 0.:
                aii = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
                aiim1 = -dt*beta[i]*ufp[i]/dxc[i]
                aiip1 = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
                field[it+1,i] = (rhs[i] - aiim1*np.roll(field[it],1)[i] - aiip1*np.roll(field[it],-1)[i])/aii
            else:
                field[it+1,i] = rhs[i]

    return field

@njit(**jitflags)
def flux_njit(Psi_L, Psi_R, U):
    """
    This function computes the upwind flux for the MPDATA scheme, based on eq (3a)-(3d)
    in: P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    Psi_L   : 1D array of floats, Psi in grid cell left of flux cell face
    Psi_R   : 1D array of floats, Psi in grid cell right of flux cell face
    U       : float or 1D array of floats with the length of Psi_L and Psi_R,
              local Courant number at cell face. U[i] is defined at the face 
              between grid cells i and i+1
    --- Output --- 
    F       : 1D array of floats describing total outgoing flux corresponding to each grid point
    """
    #U_p, U_m = np.zeros(len(U)), np.zeros(len(U))
    # Calculate U+ and U- from U
    #for i in range(len(U)):
    U_p = 0.5*(U + abs(U))
    U_m = 0.5*(U - abs(U))

    # Calculate the upwind flux
    F = U_p*Psi_L + U_m*Psi_R

    return F

def flux(Psi_L, Psi_R, U):
    """
    This function computes the upwind flux for the MPDATA scheme, based on eq (3a)-(3d)
    in: P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    Psi_L   : 1D array of floats, Psi in grid cell left of flux cell face
    Psi_R   : 1D array of floats, Psi in grid cell right of flux cell face
    U       : float or 1D array of floats with the length of Psi_L and Psi_R,
              local Courant number at cell face. U[i] is defined at the face 
              between grid cells i and i+1
    --- Output --- 
    F       : 1D array of floats describing total outgoing flux corresponding to each grid point
    """
    # Calculate U+ and U- from U
    U_p = 0.5*(U + abs(U))
    U_m = 0.5*(U - abs(U))

    # Calculate the upwind flux
    F = U_p*Psi_L + U_m*Psi_R

    return F


def d2dx2(field, dxc):
    """
    This function computes the second derivative of a field with respect to x
    using a second-order central difference scheme.
    --- Input ---
    field   : 1D array of floats, field to take the derivative of
    dxc     : 1D array of floats, spacing between cell faces
    --- Output --- 
    d2dx2   : 1D array of floats, second derivative of field with respect to x
    """
    d2dx2 = (np.roll(field,-1) - 2*field + np.roll(field,1))/(dxc*dxc)

    return d2dx2


def thirdorder(field, dxc, uf, dt):
    """This function includes the third-order spatial correction to MPDATA from Smolarkiewicz and Margolin (1998, specifically Eq.36) and Waruszewski et al. (2018).
    Input:
    field: 1D array of floats, field to advect
    dxc: 1D array of floats, spacing between cell faces
    uf: 1D array of floats, velocity defined at faces
    dt: float, timestep
    Output:
    dU: 1D array of floats, third-order spatial correction to MPDATA
    """
    # Initialisation
    G = 1 # Jacobian, 1 for a uniform grid

    # Assume uniform grid, i.e., u @cells = u @faces
    C = uf*dt/dxc # Courant number
    #TOC = - (3*C*np.absolute(C)/G - 2*C*C*C/(G*G) - C)*(np.roll(field,-2) - 2*field + np.roll(field,2))/12
    TOC = (3*C*np.absolute(C)/G - 2*C*C*C/(G*G) - C)*(2*np.roll(field,-1) - 2*np.roll(field,1) + np.roll(field,2) - np.roll(field,-2))/12

    return TOC


def LW3aiU(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1, FCT=False, returnLO=False, switch_sign=False, explFCTuntil2=False, FCTadim=False):
    """This function implements the third-order Lax-Wendroff scheme and an adaptively implicit first-order scheme. Then FCT is applied to determine how much of the LW3 scheme can be applied while ensuring boundedness.
    Assumes constant velocity and uniform grid.
    Third-order correction is based on MPDATA_LW3 derivation from Hilary from 29-07-2024. Third-order Lax-Wendroff. 
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    dt      : float, timestep
    uf      : array of floats, velocity defined at faces
    dxc     : array of floats, spacing between cell faces
    solver  : string, optional. If 'NumPy', use numpy's linalg.solve. If 'Jacobi', use Jacobi iteration.
    do_beta : string, optional. If 'switch', beta is 0 for c<=1 and 1 for c>1. If 'blend', beta is 0 for c<=1 and 1-1/c for c>1. Default is 'blend'.
    niter   : integer, optional. Number of iterations for the solver
    FCT     : boolean, optional. If True, use FCT to limit the high-order flux
    returnLO: boolean, optional. If True, return the low-order solution
    switch_sign: boolean, optional. If True, switch the sign of the second- and third-order correction. 07-11-2024 testing for diffusive effect of C>1 region.
    FCTadim : boolean, optional. If True, use beta to determine how much upwind FCT we can apply. 
    --- Output ---
    field   : 2D array of floats. Outputs each timestep of the field while advecting
    """    
    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    finalfield_FP = np.zeros(np.shape(field))
    finalfield_FP[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)

    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2
    previous = np.full(len(init), None) 

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    #chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # Low-order solution: adaptively implicit upwind/ftbs and btbs
        flx_LO = dt*flux(np.roll(field[it],1), field[it], uf)
        rhs = field[it] - (np.roll((1. - beta)*flx_LO,-1) - (1. - beta)*flx_LO)/dxc
        field_LO = solverfn(M, field[it], rhs, niter)

        # High-order solution: third-order Lax-Wendroff
        flx_FP = dt*flux(np.roll(field[it],1), field[it], uf)
        flx_SP = 0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        flx_TP = - uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2

        if c[0] > 1. and switch_sign == True: # Take care with a non-uniform grid! This assumes uniform grid and velocity.
            flx_SP = - flx_SP 
            flx_TP = - flx_TP

        cbound = 1. if explFCTuntil2 == False else 2.
        
        previous = [field[it][i] if c[i] <= cbound else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.

        corr = flx_FP + flx_SP + flx_TP - flx_LO # high-order flux - low-order flux # corr[i] defined at i-1/2
        if FCT == True:
            corr = lim.FCT(field_LO, corr, dxc, previous)#, adim=FCTadim)
        field[it+1] = field_LO - (np.roll(corr,-1) - corr)/dxc
            
        if returnLO == True:
            finalfield_FP[it+1] = field_LO.copy()

    if returnLO == True:
        return finalfield_FP
    else:
        return field
    

def FCTex_im(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1, FCT=False, returnLO=False, returnFCT=False, returnHO=False, explFCTuntil2=False):
    """This function implements a scheme that applies FCT to the explicit parts of HO and LO solution, and then applies the LO implicit part."""

    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    finalfield_LO = np.zeros(np.shape(field))
    finalfield_FCT = np.zeros(np.shape(field))
    finalfield_HO = np.zeros(np.shape(field))
    finalfield_LO[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)
    finalfield_FCT[0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)
    finalfield_HO [0] = init.copy() # setting this to be nonzero allows to compare the RMSE over time with upwind (if just a single timestep)

    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2
    previous = np.full(len(init), None) 

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        # Low-order explicit solution (adaptively implicit upwind/ftbs and btbs)
        flx_LOe = dt*(1. - beta)*flux(np.roll(field[it],1), field[it], uf)
        field_LOe = field[it] - (np.roll(flx_LOe,-1) - flx_LOe)/dxc

        cbound = 1. if explFCTuntil2 == False else 2.
        previous = [field[it][i] if c[i] <= cbound else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.

        # High-order solution: third-order Lax-Wendroff
        flx_FPe = (1. - beta)*dt*flux(np.roll(field[it],1), field[it], uf)
        flx_SPe = (1. - beta)*0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        flx_TPe = -(1. - beta)*uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2

        flx_HOe = flx_FPe + flx_SPe + flx_TPe
        field_HOe = field[it] - (np.roll(flx_HOe,-1) - flx_HOe)/dxc

        #chi = max(1 - 2*beta, 0) # chi[i] is at i-1/2 #!!!
        eta = 1.
        chi=1.5
        # High-order solution: third-order Lax-Wendroff # second choice!
        flx_FPe2 = (1. - beta)*dt*flux(np.roll(field[it],1), field[it], uf)
        flx_SPe2 = (1. - beta)*0.5*uf*dt*(1-chi*c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        flx_TPe2 = -uf*dt/6*(1-eta*c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2

        flx_HOe2 = flx_FPe2 + flx_SPe2 #+ flx_TPe2
        field_HOe2 = field[it] - (np.roll(flx_HOe2,-1) - flx_HOe2)/dxc

        # calculate HO field
        flx_FP = dt*flux(np.roll(field[it],1), field[it], uf)
        flx_SP = 0.5*uf*dt*(1-c)*(field[it] - np.roll(field[it],1)) # assumes u>0 # flx_SP[i] defined at i-1/2
        flx_TP = -uf*dt/6*(1-c*c)*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) # assumes u>0 # flx_TP[i] defined at i-1/2

        flx_HO = flx_FP + flx_SP + flx_TP
        field_HO = field[it] - (np.roll(flx_HO,-1) - flx_HO)/dxc

        corr = flx_FPe + flx_SPe + flx_TPe - flx_LOe # high-order flux - low-order flux # corr[i] defined at i-1/2
        print('field[it]', field[it])
        print('field_LOe', field_LOe)
        print('field_HOe', field_HOe)
        print('field_HOe2', field_HOe2)

        print('1-beta', 1-beta)
        plt.subplots(figsize=(10, 5))
        plt.plot(field[it], label='field[it]')
        plt.plot(field_HOe, label='field_HOe')
        plt.plot(field_HO, label='field_HO')
        #plt.plot(field_HOe2, label='field_HOe2')
        plt.legend()
        plt.title('FCTex_im: HOe and HO fields at it=1 compared to field[it=0]. dt=0.01,u=3.125,nx=40.')
        plt.savefig('FCTex_im_HOe_HO_field.png')
        plt.show() 

        if FCT == True:
            corr = lim.FCT(field_LOe, corr, dxc, previous)
        field_FCT = field_LOe - (np.roll(corr,-1) - corr)/dxc
        field[it+1] = solverfn(M, field_FCT, field_FCT, niter)

        print('field_FCT', field_FCT)
        print('field[it+1]', field[it+1])      
        print()
        if returnLO == True:
            finalfield_LO[it+1] = field_LOe.copy()

        if returnFCT == True:
            finalfield_FCT[it+1] = field_FCT.copy()

        if returnHO == True:
            finalfield_HO[it+1] = field_HOe.copy()
    
    if returnFCT == True:
        return finalfield_FCT        
    elif returnLO == True:
        return finalfield_LO
    elif returnHO == True:
        return finalfield_HO
    else:
        return field


def aiUexcorr(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1):#, FCT=False, returnLO=False, returnFCT=False, returnHO=False, explFCTuntil2=False):
    """This scheme was derived on 25-11-2024 by AW. It has adaptively implicit upwind with a third-order explicit correction (mathematically; I think)."""

    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()

    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 
    eta = 1. - 3*beta + 3*beta*beta
    chi = np.zeros(nx)

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        rhs = field[it] - c*(1 - beta)*(field[it] - np.roll(field[it],1)) \
            - 0.5*(c*c*c*(1 - beta)*beta + c - c*c*chi)*(np.roll(field[it],-1) - 2*field[it] + np.roll(field[it],1)) \
            + (0.5*(c*c*c*c*(1 - beta)*beta*beta + c*c*beta - c*c*c*chi) + (c*c*c*c*(1 - beta)*beta*chi + c - c*c*c*eta)/6) \
            *(np.roll(field[it],-1) - 3*field[it] + 3*np.roll(field[it],1) - np.roll(field[it],2)) \
            + 0.5*c*c*c*(1 - beta)*beta*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) \
            - (0.5*c*c*c*c*(1 - beta)*beta + c*c*c*c*(1 - beta)*beta*chi/6)*(field[it] - 3*np.roll(field[it],1) + 3*np.roll(field[it],2) - np.roll(field[it],3))
        field[it+1] = solverfn(M, field[it], rhs, niter)

    return field


def aiUexcorr_testing(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1):#, FCT=False, returnLO=False, returnFCT=False, returnHO=False, explFCTuntil2=False):
    """This scheme was derived on 25-11-2024 by AW. It has adaptively implicit upwind with a third-order explicit correction (mathematically; I think)."""

    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()

    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2

    beta = 1 - 1/c # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
   
    chi = 1 - 2*beta # chi[i] is at i-1/2 #!!! 
    eta = 1. - 3*beta + 3*beta*beta
    chi = np.zeros(nx)

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        rhs = field[it] - c*(1 - beta)*(field[it] - np.roll(field[it],1)) \
            - 0.5*(c*c*c*(1 - beta)*beta + c - c*c*chi)*(np.roll(field[it],-1) - 2*field[it] + np.roll(field[it],1)) \
            + (0.5*(c*c*c*c*(1 - beta)*beta*beta + c*c*beta - c*c*c*chi) + (c*c*c*c*(1 - beta)*beta*chi + c - c*c*c*eta)/6) \
            *(np.roll(field[it],-1) - 3*field[it] + 3*np.roll(field[it],1) - np.roll(field[it],2)) \
            + 0.5*c*c*c*(1 - beta)*beta*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) \
            - (0.5*c*c*c*c*(1 - beta)*beta + c*c*c*c*(1 - beta)*beta*chi/6)*(field[it] - 3*np.roll(field[it],1) + 3*np.roll(field[it],2) - np.roll(field[it],3))
        field[it+1] = solverfn(M, field[it], rhs, niter)

    return field


def aiUexcorr_adjusted20250106(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1):#, FCT=False, returnLO=False, returnFCT=False, returnHO=False, explFCTuntil2=False):
    """This scheme was derived on 25-11-2024 by AW (06-01-2025: perhaps incorrectly, hence testing an adjusted version). It has adaptively implicit upwind with a third-order explicit correction (mathematically; I think)."""

    print('initial', init)
    print()
    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    print('C', c)
    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = 1. - 2.*beta#np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 
    eta = 1. - 3*beta + 3*beta*beta
    #chi = np.zeros(nx)
    #print('chi', chi)
    #print('beta', beta)
    #eta = np.full(len(init), -2.)
    #print('eta', eta)

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        rhs = field[it] - c*(1 - beta)*(field[it] - np.roll(field[it],1)) \
            - 0.5*(-c*c*c*(1 - beta)*beta + c - c*c*chi)*(np.roll(field[it],-1) - 2*field[it] + np.roll(field[it],1)) \
            + (0.5*(-c*c*c*c*(1 - beta)*beta*beta + c*c*beta - c*c*c*chi) + (-c*c*c*c*(1 - beta)*beta*chi + c - c*c*c*eta)/6) \
            *(np.roll(field[it],-1) - 3*field[it] + 3*np.roll(field[it],1) - np.roll(field[it],2)) \
            - 0.5*c*c*c*(1 - beta)*beta*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2)) \
            + (0.5*c*c*c*c*(1 - beta)*beta*beta + c*c*c*c*(1 - beta)*beta*chi/6)*(field[it] - 3*np.roll(field[it],1) + 3*np.roll(field[it],2) - np.roll(field[it],3))
        field[it+1] = solverfn(M, field[it], rhs, niter)

    #print('it=1', field[1,:])

    return field


def aiUexcorr2(init, nt, dt, uf, dxc, solver='NumPy', do_beta='blend', niter=1):
    """This is the second-order (supposedly) version of the aiUexcorr scheme. It is based on the same principles, but only up to a second-order correction. See the handwritten derivation from 10-01-2025."""

    c = dt*uf/dxc # assumes uniform grid
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    solverfn = getattr(sv, solver)
    beta = np.ones(len(init)) # beta[i] is at i-1/2

    if do_beta == 'switch':
        beta = np.invert((np.roll(c,1) <= 1.)*(c <= 1.)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(c)), 1 - 1/c, 1 - 1/np.roll(c,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')
    chi = 1. - 2.*beta#np.maximum(1 - 2*beta, np.zeros(len(init))) # chi[i] is at i-1/2 #!!! 
    eta = 1. - 3*beta + 3*beta*beta

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
        M[i,(i-1)%len(init)] = -dt*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%len(init)] = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
    
    # Time stepping
    for it in range(nt):
        rhs = field[it] - c*(1 - beta)*(field[it] - np.roll(field[it],1)) \
            - 0.5*(-c*c*c*(1 - beta)*beta + c - c*c*chi)*(np.roll(field[it],-1) - 2*field[it] + np.roll(field[it],1)) \
            - 0.5*c*c*c*(1 - beta)*beta*(field[it] - 2*np.roll(field[it],1) + np.roll(field[it],2))
        field[it+1] = solverfn(M, field[it], rhs, niter)

    return field


def RK2QC(init, nt, dt, uf, dxc, solver='NumPy', kmax=2, set_alpha='max', set_beta=None):
    """This scheme solves second-order Runge-Kutta quasi-cubic scheme. See HW notes sent on 27-11-2024."""
    # assumes u>0

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    fieldh_HO_n, flx_HO_n, fieldh_1st_km1, flx_1st_km1, fieldh_HO_km1, fieldh_HOC_km1, flx_HOC_km1, rhs, beta = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)
    M = np.zeros((nx, nx))
    solverfn = getattr(sv, solver)

    c = dt*uf/dxc # assumes uniform grid
    #alpha = np.maximum(0.5, 1. - 1./c) # assumes uniform grid # alpha[i] is at i-1/2
    for i in range(nx):
        beta[i] = 0. if c[i] < 0.8 else 1 # beta[i] is at i-1/2
    #print(beta)
    if set_beta == 'one': 
        beta = np.ones(len(init))

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    if set_alpha == 'max': # assumes uniform grid # alpha[i] is at i-1/2
        alpha = np.maximum(0.5, 1. - 1./c)
    elif set_alpha == 'half':
        alpha = np.full(len(init), 0.5)
    else:
        print('Error: set_alpha must be either "half" or "max"')

    for i in range(nx):	# includes flx_1st_k # based on implicit upwind matrices above.
        M[i,i] = 1. + dt*(np.roll(alpha*beta*ufp,-1)[i] - alpha[i]*beta[i]*ufm[i])/dxc[i]
        M[i,i-1] = -dt*alpha[i]*beta[i]*ufp[i]/dxc[i]
        M[i,(i+1)%nx] = dt*np.roll(alpha*beta*ufm,-1)[i]/dxc[i]

    for it in range(nt):
        field[it+1] = field[it].copy() # not actually in the equations, this is to make the computer code more concise
        for k in range(kmax):
            for i in range(nx):
                fieldh_HO_n[i] = sd.quadh(field[it,i-2], field[it,i-1], field[it,i]) # [i] defined at i-1/2
                flx_HO_n[i] = (1. - alpha[i])*uf[i]*fieldh_HO_n[i] # [i] defined at i-1/2
                fieldh_1st_km1[i] = field[it+1,i-1] # upwind # [i] defined at i-1/2 # not actually field[it+1], this is to make the computer code more concise
                flx_1st_km1[i] = alpha[i]*(1. - beta[i])*uf[i]*fieldh_1st_km1[i] # [i] defined at i-1/2
                fieldh_HO_km1[i] = sd.quadh(field[it+1,i-2], field[it+1,i-1], field[it+1,i]) # [i] defined at i-1/2
                fieldh_HOC_km1[i] = fieldh_HO_km1[i] - fieldh_1st_km1[i] #fieldh_HO_n[i] - fieldh_1st_km1[i] # [i] defined at i-1/2
                flx_HOC_km1[i] = alpha[i]*uf[i]*fieldh_HOC_km1[i] # [i] defined at i-1/2
            for i in range(nx):
                rhs[i] = field[it,i] - dt*(ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i]) + \
                        ddx(flx_1st_km1[i], flx_1st_km1[(i+1)%nx], dxc[i]) + \
                        ddx(flx_HOC_km1[i], flx_HOC_km1[(i+1)%nx], dxc[i])) # [i] defined at i
            field[it+1] = solverfn(M, fieldh_1st_km1, rhs, 1)    

    return field


def RK2QC_noPC(init, nt, dt, uf, dxc, solver='NumPy', set_alpha='max', FCT=False, nonnegative=False, doubleFCT=False, doubleFCT_noupdate=False, FCTnonneg=False):
    """This scheme solves the non-predictor-corrector version (see above for that one) of the RK2_QC scheme. See HW notes sent on 27-11-2024 and AW notes 14-01-2025.
    alpha = 'half' or 'max' (0.5 or max(0.5,1-1/c))
    """
    # assumes u>0

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    M = np.zeros((nx, nx))
    flx_HO_n, rhs, au = np.zeros(nx), np.zeros(nx), np.zeros(nx)

    c = dt*uf/dxc # assumes uniform grid
    solverfn = getattr(sv, solver)

    if set_alpha == 'max': # assumes uniform grid # alpha[i] is at i-1/2
        alpha = np.maximum(0.5, 1. - 1./c)
    elif set_alpha == 'half':
        alpha = np.full(len(init), 0.5)
    else:
        print('Error: set_alpha must be either "half" or "max"')

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))
    
    au = alpha*ufp # assumes u>0, i defined at i-1/2

    for i in range(nx):
        M[i,i] = 1. + dt*(5.*np.roll(au,-1)[i] - 2.*au[i])/(6.*dxc[i])
        M[i,i-1] = -dt*(np.roll(au,-1)[i] + 5.*au[i])/(6.*dxc[i])
        M[i,(i-2)] = dt*au[i]/(6.*dxc[i])   
        M[i,(i+1)%nx] = dt*np.roll(au,-1)[i]/(3.*dxc[i])
        
    for it in range(nt):
        for i in range(nx):
            flx_HO_n[i] = dt*(1 - alpha[i])*uf[i]*sd.quadh(field[it,i-2], field[it,i-1], field[it,i]) # [i] defined at i-1/2
        for i in range(nx):
            rhs[i] = field[it,i] - (ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i])) # [i] defined at i
        field[it+1] = solverfn(M, field[it], rhs, 1) # 15-01-2025: probably only 'NumPy' works here properly as the matrix is not as sparse as before

        if nonnegative == True:
            flx_HO_np1 = dt*uf*(alpha*sd.quadh(np.roll(field[it+1],2), np.roll(field[it+1],1), field[it+1])) # assumes u>0, [i] defined at i-1/2
            flx_HO = flx_HO_n + flx_HO_np1 # [i] defined at i-1/2
            field[it+1] = lim.nonneg(field[it+1], flx_HO, dxc)

        if FCT == True or doubleFCT == True or doubleFCT_noupdate == True or FCTnonneg == True: # FCT: low-order solution is adaptively implicit upwind. (assumes u>0 so actually FTBS&BTBS) (doubleFCT = FCT applied twice)
            if FCT == True:
                FCTfunc = getattr(lim, 'FCT')
            elif doubleFCT == True:
                FCTfunc = getattr(lim, 'doubleFCT')
            elif doubleFCT_noupdate == True:
                FCTfunc = getattr(lim, 'doubleFCT_noupdate')
            elif FCTnonneg == True:
                FCTfunc = getattr(lim, 'FCTnonneg')
        
            # Calculate low-order solution field_LO
            M_LO = np.zeros((nx, nx)) 
            theta = np.maximum(1. - 1./c, np.zeros(nx)) # [i] defined at i-1/2; theta: blend! Off-centring in time for aiUpwind
            for i in prange(nx):
                M_LO[i,i] = 1. + theta[i]*c[i]
                M_LO[i,(i-1)%len(init)] = -1.*theta[i]*c[i]
            rhs_LO = field[it] - c*(1. - theta)*(field[it] - np.roll(field[it],1))
            field_LO = solverfn(M_LO, field[it], rhs_LO, 1)
            
            # Calculate low-order fluxes flx_LO
            flx_LO = dt*uf*(theta*np.roll(field_LO,1) + (1. - theta)*np.roll(field[it],1)) # assumes u>0, [i] defined at i-1/2
            
            # Calculate high-order fluxes flx_HO
            flx_HO_np1 = dt*uf*(alpha*sd.quadh(np.roll(field[it+1],2), np.roll(field[it+1],1), field[it+1])) # assumes u>0, [i] defined at i-1/2
            flx_HO = flx_HO_n + flx_HO_np1 # [i] defined at i-1/2
            
            # Apply flux-corrected transport
            previous = [field[it][i] if c[i] <= 1. else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.
            corr = flx_HO - flx_LO # flux between HO and LO, [i] defined at i-1/2
            corr = FCTfunc(field_LO, corr, dxc, previous)
            field[it+1] = field_LO - (np.roll(corr,-1) - corr)/dxc
        
    return field


def RK2QC_noPCiter(init, nt, dt, uf, dxc, solver='NumPy', kmax=2, set_alpha='max', set_beta='one'):
    """This scheme solves an RK2QC scheme that is similar to the RK2QC and RK2QC_noPC schemes above. It is Eq.(99) in SLBweek20250109.pdf.
    beta is by default set to 1 independent of the Courant number, but can be set to 'var' to become 0 for C<0.8.
    Difference with RK2QC_noPC above: we iterate over the scheme twice, where the alpha terms are now iterated over: n and n+1 become k-1 and k for k=1,2. See the difference between equation Eq.(86) and (99) in SLBweek20250109.pdf"""
    # assumes u>0

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    M = np.zeros((nx, nx))
    flx_HO_n, flx_HO_km1, rhs, abu = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)

    c = dt*uf/dxc # assumes uniform grid
    solverfn = getattr(sv, solver)
    if set_beta == 'one': # beta[i] is at i-1/2
        beta = np.ones(len(init))
    elif set_beta == 'var':
        beta = np.zeros(len(init))
        for i in range(nx):
            beta[i] = 0. if c[i] < 0.8 else 1
    else:
        print('Error: set_beta must be either "one" or "var"')
    if set_alpha == 'max': # assumes uniform grid # alpha[i] is at i-1/2
        alpha = np.maximum(0.5, 1. - 1./c)
    elif set_alpha == 'half':
        alpha = np.full(len(init), 0.5)
    else:
        print('Error: set_alpha must be either "half" or "max"')

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))
    
    for i in range(nx):
        abu[i] = alpha[i]*beta[i]*ufp[i] # assumes u>0, i defined at i-1/2

    for i in range(nx):
        M[i,i] = 1. + dt*(5.*np.roll(abu,-1)[i] - 2.*abu[i])/(6.*dxc[i])
        M[i,i-1] = -dt*(np.roll(abu,-1)[i] + 5.*abu[i])/(6.*dxc[i])
        M[i,(i-2)] = dt*abu[i]/(6.*dxc[i])   
        M[i,(i+1)%nx] = dt*np.roll(abu,-1)[i]/(3.*dxc[i])

    for it in range(nt):
        field[it+1] = field[it].copy() # not in eq, for code simplification
        for k in range(kmax):
            for i in range(nx):
                flx_HO_n[i] = (1 - alpha[i])*uf[i]*sd.quadh(field[it,i-2], field[it,i-1], field[it,i]) # [i] defined at i-1/2
                flx_HO_km1[i] = alpha[i]*(1 - beta[i])*uf[i]*sd.quadh(field[it+1,i-2], field[it+1,i-1], field[it+1,i]) # not it+1 in eq, for code simplification
            for i in range(nx):
                rhs[i] = field[it,i] - dt*(ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i]) + ddx(flx_HO_km1[i], flx_HO_km1[(i+1)%nx], dxc[i])) # [i] defined at i
            field[it+1] = solverfn(M, field[it], rhs, 1) # 15-01-2025: probably only 'NumPy' works here properly as the matrix is not as sparse as before

    return field


def ddx(fmh, fph, dxc):
    """This function computes the first derivative of a field f with respect to x with a finite-volume method. 
    dfdx[i] = (f_{i+1/2} - f_{i-1/2})/dx
    fmh : f_{i-1/2}
    fph : f_{i+1/2}
    """
    return (fph - fmh)/dxc


def PR05TVr():
    gamma = 1. - 0.5*np.sqrt(2)
    A = np.array([[gamma, 0, 0],[1 - 2*gamma, gamma, 0],[0.5 - gamma, 0, gamma]])
    b = np.array([1/6, 1/6, 2/3])
    return A, b


def PR05TVl():
    A = np.array([[0, 0, 0],[1, 0, 0],[0.25, 0.25, 0]])
    b = np.array([1/6, 1/6, 2/3])
    return A, b


def ImRK3QC(init, nt, dt, uf, dxc, butcher=PR05TVr, solver='NumPy'):
    """
    This scheme implements IRK3 from Pareschi and Russo 2005 Table V right Butcher tableau, combined with the QC spatial discretisation also used in the RK2QC scheme.
    Assumes uniform grid
    """
    A, b = butcher()
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    solverfn = getattr(sv, solver)
    M = np.zeros((nx, nx))

    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!!!)
        M[i,i] = 1. + dt*A[0,0]*(5.*np.roll(uf,-1)[i] - 2.*uf[i])/(6.*dxc[i]) 
        M[i,i-1] = -dt*A[0,0]*(np.roll(uf,-1)[i] + 5.*uf[i])/(6.*dxc[i])
        M[i,(i-2)] = dt*A[0,0]*uf[i]/(6.*dxc[i])   
        M[i,(i+1)%nx] = dt*A[0,0]*np.roll(uf,-1)[i]/(3.*dxc[i])

    for it in range(nt): # assumes u>0
        field_k1 = solverfn(M, field[it], field[it], 1)
        flx_k1 = sd.quadh(np.roll(field_k1,2), np.roll(field_k1,1), field_k1) # [i] defined at i-1/2
        rhs_k2 = field[it] - dt*uf*A[1,0]*ddx(flx_k1, np.roll(flx_k1,-1), dxc) # assumes u>0
        field_k2 = solverfn(M, field[it], rhs_k2, 1)
        flx_k2 = sd.quadh(np.roll(field_k2,2), np.roll(field_k2,1), field_k2) # [i] defined at i-1/2
        rhs_k3 = field[it] - dt*uf*A[2,0]*ddx(flx_k1, np.roll(flx_k1,-1), dxc) # assumes u>0
        field_k3 = solverfn(M, field[it], rhs_k3, 1)
        flx_k3 = sd.quadh(np.roll(field_k3,2), np.roll(field_k3,1), field_k3) # [i] defined at i-1/2
        field[it+1] = field[it] - dt*b[0]*uf*ddx(flx_k1, np.roll(flx_k1,-1), dxc) \
            - dt*b[1]*uf*ddx(flx_k2, np.roll(flx_k2,-1), dxc) \
            - dt*b[2]*uf*ddx(flx_k3, np.roll(flx_k3,-1), dxc) # assumes u>0

    return field


def RK3QC(init, nt, dt, uf, dxc, butcher=PR05TVl, solver='NumPy'):
    """
    This scheme implements RK3 from Pareschi and Russo 2005 Table V left Butcher tableau, combined with the QC spatial discretisation also used in the RK2QC scheme.
    Assumes uniform grid
    """
    A, b = butcher()
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    solverfn = getattr(sv, solver)
    M = np.zeros((nx, nx))

    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!!!)
        M[i,i] = 1. + dt*A[0,0]*(5.*np.roll(uf,-1)[i] - 2.*uf[i])/(6.*dxc[i]) 
        M[i,i-1] = -dt*A[0,0]*(np.roll(uf,-1)[i] + 5.*uf[i])/(6.*dxc[i])
        M[i,(i-2)] = dt*A[0,0]*uf[i]/(6.*dxc[i])   
        M[i,(i+1)%nx] = dt*A[0,0]*np.roll(uf,-1)[i]/(3.*dxc[i])

    for it in range(nt): # assumes u>0
        field_k1 = solverfn(M, field[it], field[it], 1)
        flx_k1 = sd.quadh(np.roll(field_k1,2), np.roll(field_k1,1), field_k1) # [i] defined at i-1/2
        rhs_k2 = field[it] - dt*uf*A[1,0]*ddx(flx_k1, np.roll(flx_k1,-1), dxc) # assumes u>0
        field_k2 = solverfn(M, field[it], rhs_k2, 1)
        flx_k2 = sd.quadh(np.roll(field_k2,2), np.roll(field_k2,1), field_k2) # [i] defined at i-1/2
        rhs_k3 = field[it] - dt*uf*A[2,0]*ddx(flx_k1, np.roll(flx_k1,-1), dxc) - dt*uf*A[2,1]*ddx(flx_k2, np.roll(flx_k2,-1), dxc) # assumes u>0
        field_k3 = solverfn(M, field[it], rhs_k3, 1)
        flx_k3 = sd.quadh(np.roll(field_k3,2), np.roll(field_k3,1), field_k3) # [i] defined at i-1/2
        field[it+1] = field[it] - dt*b[0]*uf*ddx(flx_k1, np.roll(flx_k1,-1), dxc) \
            - dt*b[1]*uf*ddx(flx_k2, np.roll(flx_k2,-1), dxc) \
            - dt*b[2]*uf*ddx(flx_k3, np.roll(flx_k3,-1), dxc) # assumes u>0

    return field


def lExrImRK3QC(init, nt, dt, uf, dxc, butcherIm=PR05TVr, butcherEx=PR05TVl, solver='NumPy'):
    """
    This scheme implements the Pareschi and Russo 2005 Table V left and right Butcher tableaus, combined with the QC spatial discretisation also used in the RK2QC scheme.
    Assumes uniform grid. The default is to implement the explicit tableau in the left half of the spatial domain and the implicit tableau in the right half of the spatial domain. 
    """
    AIm, bIm = butcherIm()
    AEx, bEx = butcherEx()
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    solverfn = getattr(sv, solver)
    M = np.zeros((nx, nx))

    # Setting the left half of the domain to be explicit (beta=0) and the right half implicit (beta=1)
    beta = np.ones(len(init)) # beta[i] is at i-1/2
    for i in range(int(nx/2)):
        beta[i] = 0. 

    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!!!) # this also assumes that the implicit/explicit regions don't change throughout the simulation
        M[i,(i-2)] = dt*AIm[0,0]*beta[i]*uf[i]/(6.*dxc[i])   
        M[i,i-1] = -dt*AIm[0,0]*(np.roll(beta*uf,-1)[i] + 5.*beta[i]*uf[i])/(6.*dxc[i])
        M[i,i] = 1. + dt*AIm[0,0]*(5.*np.roll(beta*uf,-1)[i] - 2.*beta[i]*uf[i])/(6.*dxc[i]) 
        M[i,(i+1)%nx] = dt*AIm[0,0]*np.roll(beta*uf,-1)[i]/(3.*dxc[i])

    for it in range(nt): # assumes u>0
        field_k1 = solverfn(M, field[it], field[it], 1)
        
        flx_k1 = sd.quadh(np.roll(field_k1,2), np.roll(field_k1,1), field_k1) # [i] defined at i-1/2
        rhs_k2_Im = - dt*uf*AIm[1,0]*ddx(beta*flx_k1, np.roll(beta*flx_k1,-1), dxc)
        rhs_k2_Ex = - dt*uf*AEx[1,0]*ddx((1-beta)*flx_k1, np.roll((1-beta)*flx_k1,-1), dxc)
        rhs_k2 = field[it] + rhs_k2_Im + rhs_k2_Ex # assumes u>0
        field_k2 = solverfn(M, field[it], rhs_k2, 1)
        
        flx_k2 = sd.quadh(np.roll(field_k2,2), np.roll(field_k2,1), field_k2) # [i] defined at i-1/2
        rhs_k3_Im = - dt*uf*AIm[2,0]*ddx(beta*flx_k1, np.roll(beta*flx_k1,-1), dxc) - dt*uf*AIm[2,1]*ddx(beta*flx_k2, np.roll(beta*flx_k2,-1), dxc)
        rhs_k3_Ex = - dt*uf*AEx[2,0]*ddx((1-beta)*flx_k1, np.roll((1-beta)*flx_k1,-1), dxc) - dt*uf*AEx[2,1]*ddx((1-beta)*flx_k2, np.roll((1-beta)*flx_k2,-1), dxc)
        rhs_k3 = field[it] + rhs_k3_Im + rhs_k3_Ex  # assumes u>0
        field_k3 = solverfn(M, field[it], rhs_k3, 1)
        
        flx_k3 = sd.quadh(np.roll(field_k3,2), np.roll(field_k3,1), field_k3) # [i] defined at i-1/2
        rhs_final_Im =  - dt*bIm[0]*uf*ddx(beta*flx_k1, np.roll(beta*flx_k1,-1), dxc) \
            - dt*bIm[1]*uf*ddx(beta*flx_k2, np.roll(beta*flx_k2,-1), dxc) \
            - dt*bIm[2]*uf*ddx(beta*flx_k3, np.roll(beta*flx_k3,-1), dxc) # assumes u>0
        rhs_final_Ex = - dt*bEx[0]*uf*ddx((1-beta)*flx_k1, np.roll((1-beta)*flx_k1,-1), dxc) \
            - dt*bEx[1]*uf*ddx((1-beta)*flx_k2, np.roll((1-beta)*flx_k2,-1), dxc) \
            - dt*bEx[2]*uf*ddx((1-beta)*flx_k3, np.roll((1-beta)*flx_k3,-1), dxc) # assumes u>0
        field[it+1] = field[it] + rhs_final_Im + rhs_final_Ex

    return field


def PPM(init, nt, dt, uf, dxc, MULES=False, nIter=1, iterFCT=False):
    """This scheme implements the piecewise parabolic method (PPM) by Colella and Woodward 1984. See HW notes MULES vs FCT 31-01-2025. Also works for large Courant numbers."""

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc # assumes uniform grid
    intc = c.astype(int)
    dc = c - intc.astype(float)
    fieldh, fieldprime, field6, ksum = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)

    for it in range(nt):
        fieldprime = 7./12.*(np.roll(field[it],1) + field[it]) - 1./12.*(np.roll(field[it],-1) + np.roll(field[it],2)) # [i] defined at i-1/2
        field6 = 6.*(field[it] - 0.5*(np.roll(fieldprime,-1) + fieldprime)) # [i] defined at i
        dfield = np.roll(fieldprime, -1) - fieldprime # [i] defined at i
        fieldh_dc = fieldprime - 0.5*np.roll(dc,1)*(np.roll(dfield,1) - (1. - 2./3.*np.roll(dc,1))*np.roll(field6,1)) # [i] defined at i-1/2
        for j in range(nx):
            ksum[j] = 0.
            for k in range(j - intc[j] + 1, j + 1):
                ksum[j] += field[it,k%nx] # [j] defined at j+1/2
        fieldh = 1./np.roll(c,1)*np.roll(ksum,1) + np.roll(dc,1)/np.roll(c,1)*np.roll(fieldh_dc,intc[0]) # assumes uniform c # [i] defined at i-1/2
        if MULES == True:
            fieldh = lim.MULES(field[it], fieldh, c, nIter=nIter)
        if iterFCT == True:
            previous = [field[it][i] if c[i] <= 1. else None for i in range(len(init))] # determines whether FCT also uses field[it] for bounds. If an element is None, it is not used.
            field[it+1] = lim.iterFCT(fieldh, dxc, dt, uf, c, field[it], previous, niter=nIter)
        else:
            field[it+1] = field[it] - c*(np.roll(fieldh,-1) - fieldh)
            

    return field


def butcherExSSP3433():
    """Left explicit Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005)."""
    A = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0], [0, 0.25, 0.25, 0]])
    b = np.array([[0, 1/6, 1/6, 2/3]])
    return A, b


def butcherImSSP3433():
    """Right implicit Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005)."""
    alpha, beta, eta = 0.24169426078821, 0.06042356519705, 0.12915286960590
    A = np.array([[alpha, 0, 0, 0], [-alpha, alpha, 0, 0], [0, 1 - alpha, alpha, 0], [beta, eta, 0.5 - beta - eta - alpha, alpha]])
    b = np.array([[0, 1/6, 1/6, 2/3]])
    return A, b


def ImSSP3QC(init, nt, dt, uf, dxc, MULES=False, nIter=1):
    """This scheme implements the timestepping from the right Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005), combined with the QC spatial discretisation also used in the RK2QC scheme. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImSSP3433()
    M = sd.QCmatrix(nx, dt, dxc, uf, A[0,0]) # assumes the matrix is the same for the multiple solves, i.e., A[0,0] = A[1,1] = A[2,2] = A[3,3] (not always true)
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc # assumes uniform grid

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.quadh(np.roll(field_k,2), np.roll(field_k,1), field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def test_IRK3QC_loops(init, nt, dt, uf, dxc): # 12-02-2025: indeed matches IRK3QC above, as it should
    """This scheme implements the timestepping from the right Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005), combined with the QC spatial discretisation also used in the RK2QC scheme. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = PR05TVr()
    M = sd.QCmatrix(nx, dt, dxc, uf, A[0,0]) # assumes the matrix is the same for the multiple solves, i.e., A[0,0] = A[1,1] = A[2,2] = A[3,3] (not always true)
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    print()
    for it in range(nt):
        field_k = field[it].copy()
        for ik in range(len(b)):
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.quadh(np.roll(field_k,2), np.roll(field_k,1), field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
        field[it+1] = field[it] + dt*np.dot(b, f)
    return field


def butcherExARS3233():
    """Left explicit Butcher tableau of the ARS3(2,3,3) scheme (see Weller, Lock, Wood 2013 - note the b definition is incorrect in that paper)."""
    gamma = 0.5 + np.sqrt(3)/6
    A = np.array([[0, 0, 0], [gamma, 0, 0], [gamma - 1, 2*(1 - gamma), 0]])
    b = np.array([[0, 0.5, 0.5]])
    return A, b


def butcherImARS3233():
    """Right implicit Butcher tableau of the ARS3(2,3,3) scheme (see Weller, Lock, Wood 2013 - note the b definition is incorrect in that paper)."""
    gamma = 0.5 + np.sqrt(3)/6
    A = np.array([[0, 0, 0], [0, gamma, 0], [0, 1 - 2*gamma, gamma]])
    b = np.array([[0, 0.5, 0.5]])
    return A, b


def ImARS3QC(init, nt, dt, uf, dxc, MULES=False, nIter=1):
    """This scheme implements the timestepping from the right Butcher tableau of the ARS3(2,3,3) scheme (see Weller, Lock, Wood 2013), combined with the QC spatial discretisation also used in the RK2QC scheme. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImARS3233()
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc # assumes uniform grid

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = sd.QCmatrix(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.quadh(np.roll(field_k,2), np.roll(field_k,1), field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImSSP3C4(init, nt, dt, uf, dxc, MULES=False, nIter=1): # I tested the accuracy of changing the setup for MULES and that didn't changed the results (up to a rounding error many digits after the comma)
    """This scheme implements the timestepping from the right Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005), combined with the centred fourth order spatial discretisation. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImSSP3433()
    M = sd.Mfourth22(nx, dt, dxc, uf, A[0,0]) # assumes the matrix is the same for the multiple solves, i.e., A[0,0] = A[1,1] = A[2,2] = A[3,3] (not always true)
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.fourth22(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImARS3C4(init, nt, dt, uf, dxc, MULES=False, nIter=1):
    """This scheme implements the timestepping from the right Butcher tableau of the ARS3(2,3,3) scheme (see Weller, Lock, Wood 2013), combined with the fourth order centred spatial discretisation. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImARS3233()
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = sd.Mfourth22(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.fourth22(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def butcherExUJ31e32():
    """Left (explicit) butcher tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme."""   
    A = np.array([[0., 0., 0., 0., 0.],[0., 0., 0., 0., 0.],[0., 1., 0., 0., 0.],[0., 0.25, 0.25, 0., 0.],[0., 1/6, 1/6, 2/3, 0.]])
    b = np.array([[0., 1/6, 1/6, 2/3, 0.]])
    return A, b


def butcherImUJ31e32():
    """Right (implicit) butcher tableau from Ullrich and Jablonowski 2012. See the Weller Lock and Wood (2013) UJ3(1+e,3,2) scheme."""   
    A = np.array([[0., 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.],[0.5, 0., 0., 0., 0.5]])
    b = np.array([[0.5, 0., 0., 0., 0.5]])
    return A, b


def ImExUJ3QC(init, nt, dt, uf, dxc, MULES=False, nIter=1):
    """This scheme implements the timestepping from the Double butcher tableau from Ullrich and Jablonowski 2012, combined with the QC spatial discretisation also used in the RK2QC scheme. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc
    A, b = butcherImUJ31e32() if c[0] > 1.6 else butcherExUJ31e32() # assumes constant c. # I initially set this (1.6) to 1 - might need to be changed to another value for stability
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = sd.QCmatrix(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.quadh(np.roll(field_k,2), np.roll(field_k,1), field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExUJ3C4(init, nt, dt, uf, dxc, MULES=False, nIter=1):
    """This scheme implements the timestepping from the double butcher tableau from Ullrich and Jablonowski 2012, combined with the fourth order centred spatial discretisation. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc
    A, b = butcherImUJ31e32() if c[0] > 1.6 else butcherExUJ31e32() # assumes constant c. # I initially set this (1.6) to 1 - might need to be changed to another value for stability
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = sd.Mfourth22(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = sd.fourth22(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExUJ3_old(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22'):
    """This scheme implements the timestepping from the double butcher tableau from Ullrich and Jablonowski 2012, combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    # SD: spatial discretisation, default is centered fourth order, i.e. fourth22
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc
    A, b = butcherImUJ31e32() if c[0] > 1.6 else butcherExUJ31e32() # assumes constant c. # I initially set this (1.6) to 1 - might need to be changed to another value for stability
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = matrix(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExUJ3(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22', butcherIm=butcherImUJ31e32, butcherEx=butcherExUJ31e32):
    """This scheme implements the timestepping from the double butcher tableau from Ullrich and Jablonowski 2012, combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    # SD: spatial discretisation, default is centered fourth order, i.e. fourth22
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc

    # Setting the off-centring in time     
    beta = np.zeros(nx) # beta[i] is at i-1/2
    for i in range(nx):
        if c[i] > 1.6: # I initially set this (1.6) to 1 - might need to be changed to another value for stability
            beta[i] = 1

    AIm, bIm = butcherIm()
    AEx, bEx = butcherEx() 
    nstages = len(bIm)
    flx, f = np.zeros((nstages, nx)), np.zeros((nstages, nx))
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(nstages):
            M = matrix(nx, dt, dxc, beta*uf, AIm[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], (1 - beta[:])*f[:ik,:]) + dt*np.dot(AIm[ik,:ik], beta[:]*f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*bIm[ik]*beta + flx[ik,:]*bEx[ik]*(1 - beta)
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImSSP3(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22'): # I tested the accuracy of changing the setup for MULES and that didn't changed the results (up to a rounding error many digits after the comma)
    """This scheme implements the timestepping from the right Butcher tableau of the SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005), combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImSSP3433()
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)
    M = matrix(nx, dt, dxc, uf, A[0,0]) # assumes the matrix is the same for the multiple solves, i.e., A[0,0] = A[1,1] = A[2,2] = A[3,3] (not always true)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExSSP3(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22', butcherIm=butcherImSSP3433, butcherEx=butcherExSSP3433):
    """This scheme implements the timestepping from the double butcher tableau from SSP3(4,3,3) scheme (Table VI from Pareschi and Russo 2005), combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    # SD: spatial discretisation, default is centered fourth order, i.e. fourth22
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc

    # Setting the off-centring in time     
    beta = np.zeros(nx) # beta[i] is at i-1/2
    for i in range(nx):
        if c[i] > 0.7: # I initially set this (1.6) to 1 - might need to be changed to another value for stability
            beta[i] = 1

    AIm, bIm = butcherIm()
    AEx, bEx = butcherEx() 
    nstages = len(bIm)
    flx, f = np.zeros((nstages, nx)), np.zeros((nstages, nx))
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(nstages):
            M = matrix(nx, dt, dxc, beta*uf, AIm[ik,ik])
            rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], (1 - beta[:])*f[:ik,:]) + dt*np.dot(AIm[ik,:ik], beta[:]*f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*bIm[ik]*beta + flx[ik,:]*bEx[ik]*(1 - beta)
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImARS3(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22'):
    """This scheme implements the timestepping from the right Butcher tableau of the ARS3(2,3,3) scheme (see Weller, Lock, Wood 2013), combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    A, b = butcherImARS3233()
    flx, f = np.zeros((len(b), nx)), np.zeros((len(b), nx))
    c = dt*uf/dxc
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(len(b)):
            M = matrix(nx, dt, dxc, uf, A[ik,ik]) # Note that for this RK scheme the diagonal elements are not all the same!
            rhs_k = field[it] + dt*np.dot(A[ik,:ik], f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*b[ik]
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExARS3(init, nt, dt, uf, dxc, MULES=False, nIter=1, SD='fourth22', butcherIm=butcherImARS3233, butcherEx=butcherExARS3233):
    """This scheme implements the timestepping from the double butcher tableau from ARS3(2,3,3) scheme (see Weller Lock Wood 2013), combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant."""
    # SD: spatial discretisation, default is centered fourth order, i.e. fourth22
    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    c = dt*uf/dxc

    # Setting the off-centring in time     
    beta = np.zeros(nx) # beta[i] is at i-1/2
    for i in range(nx):
        if c[i] > 1.2: # I initially set this (1.6) to 1 - might need to be changed to another value for stability
            beta[i] = 1

    AIm, bIm = butcherIm()
    AEx, bEx = butcherEx() 
    nstages = len(bIm)
    flx, f = np.zeros((nstages, nx)), np.zeros((nstages, nx))
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)

    for it in range(nt):
        field_k = field[it].copy()
        flx_HO = np.zeros(nx)
        for ik in range(nstages):
            M = matrix(nx, dt, dxc, beta*uf, AIm[ik,ik])
            rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], (1 - beta[:])*f[:ik,:]) + dt*np.dot(AIm[ik,:ik], beta[:]*f[:ik,:])
            field_k = np.linalg.solve(M, rhs_k)
            flx[ik,:] = fluxfn(field_k) # [i] defined at i-1/2
            f[ik,:] = -uf*ddx(flx[ik,:], np.roll(flx[ik,:],-1), dxc)
            flx_HO += flx[ik,:]*bIm[ik]*beta + flx[ik,:]*bEx[ik]*(1 - beta)
        if MULES == True:
            flx_HO = lim.MULES(field[it], flx_HO, c, nIter=nIter)
        field[it+1] = field[it] - uf*dt*ddx(flx_HO, np.roll(flx_HO,-1), dxc)

    return field


def ImExRK(init, nt, dt, uf, dxc, u_setting, MULES=False, nIter=1, SD='fourth22', RK='UJ31e32', blend='off', clim=1.6, HRES=None, AdImEx=None, output_substages=False, iterFCT=False, FCT=False, FCT_HW=False): # !!! add option for non uconstant in TIME to be recalculated every time step
    """This scheme implements the timestepping from the double butcher tableau defined with RK, combined with various (default: the fourth order centred) spatial discretisations. Assumes u>0 constant.
    
    21-04-2025: uf is probably just the first value of the velocity field if it changes in time. If the velocity changes in time, we need to recalculate the u, c and beta every time step. If the velocity is constant in space and time or only varies in space, we can use uf throughout the time stepping, without need to reculculate it every time step and for intermediate stages within a RK time step.
    - haven't tested but probably only want to use the output_substages option with nt=1
    SD: spatial discretisation, default is centered fourth order, i.e. fourth22"""

    nx = len(init)
    field = np.zeros((nt+1, nx))
    matrix = getattr(sd, 'M' + SD)
    fluxfn = getattr(sd, SD)
    field[0] = init.copy()
    xf = np.zeros(nx) 
    for i in range(len(dxc)-1): # assumes uniform grid
        xf[i+1] = xf[i] + dxc[i]
    cf = uf*dt/dxc # [i] at i-1/2
    cc_out = 0.5*(np.abs(uf) - uf + np.abs(np.roll(uf,-1)) + np.roll(uf,-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *outward* pointing velocities
    cc_in = 0.5*(np.abs(uf) + uf + np.abs(np.roll(uf,-1)) - np.roll(uf,-1))*dt/dxc # [i] at i, Courant defined at cell centers based on the *inward* pointing velocities
    betac_out = np.maximum(0., 1.-1./cc_out)
    betac_in = np.maximum(0., 1.-1./cc_in)
    beta = np.maximum(np.maximum(betac_out, np.roll(betac_out,1)), np.maximum(betac_in, np.roll(betac_in, 1))) # [i] at i-1/2

    AIm, bIm = globals()['butcherIm' + RK]()
    cIm = AIm.sum(axis=1)
    AEx, bEx = globals()['butcherEx' + RK]() 
    cEx = AEx.sum(axis=1)
    nstages = np.shape(bIm)[1] # I changed bIm to be multidimensional - len(bIm) won't work properly anymore in this case. (i.e. other ImEx schemes won't work with the butcher functions as of 25-04-2025)
    # Resetting A to include b
    AEx = np.concatenate((AEx,bEx), axis=0)
    AIm = np.concatenate((AIm,bIm), axis=0)
    AEx = np.concatenate((AEx, np.zeros((nstages+1,1))), axis=1)
    AIm = np.concatenate((AIm, np.zeros((nstages+1,1))), axis=1)
    flx_k, fEx, fIm, flx_contribution_from_stage_k = np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx)), np.zeros((nstages+1, nx))

    if u_setting == 'varying_space_time': # 25-04-2025: NOT WORKING 
        # We are only setting this up with blend == 'sm' -> a smooth transition from 0 to 1 with 1-1/c
        # For a varying velocity field in time, for the offcentring calculation, we need to use the maximum velocity that will appear locally at a certain face from n to n+1 to calculate the c and theta. 
        for it in range(nt):
        # to do: I NEED TO RECALCULATE at different intermediate stages. -- check uf below
            betaset = False
            uf = an.velocity_varying_space_time(xf, it*nt) # recalculate velocity # where to put this?
            while betaset == False:  
                maxuf = uf.copy() 
                cf = maxuf*dt/(0.5*dxc) # recalculate Courant number # assumes uniform grid, which is the assumption in any case with the varying space time u setting
                beta = np.maximum(0., 1. - 1./cf)   
                for istage in range(nstages):
                    ufstage = an.velocity_varying_space_time(xf, it*nt + (1-beta)*cEx[istage]*dt + beta*cIm[istage]*dt) # recalculate velocity for the stage
                    maxuf = np.maximum(maxuf, ufstage)

                #if  : betaset = True
            # 21-04-2025: don't have this working yet. Not sure of the best way how to combine u and beta for stability. Have not tested varying_space_time yet. 
            #----
            #maxuf = # calculate the max velocity at the time points given by the Butcher c
            #c = dt*maxuf/dxc # !!! adjust # !!! 21-04-2025: put into the time loop? # !!! and do I want to assume a smooth beta blend for this i.e. when in time loop to avoid 
            #----
    else:       
        for it in range(nt):
            field_k = field[it].copy()
            flx_HO = np.zeros(nx)
            for ik in range(nstages+1):
                # Calculate the field at stage k
                M = matrix(nx, dt, dxc, beta*uf, AIm[ik,ik]) # [i] at i
                rhs_k = field[it] + dt*np.dot(AEx[ik,:ik], fEx[:ik,:]) + dt*np.dot(AIm[ik,:ik], fIm[:ik,:]) # [i] at i
                field_k = np.linalg.solve(M, rhs_k) # [i] at i
                if output_substages: 
                    plt.plot(xf, field_k, label='stage ' + str(ik))
                    print('k =', ik)
                    print('field_k', field_k)
                    print()
                # Calculate the flux based on the field at stage k
                flx_k[ik,:] = uf*fluxfn(field_k) # [i] at i-1/2
                fEx[ik,:] = -ddx((1 - beta)*flx_k[ik,:], np.roll((1 - beta)*flx_k[ik,:],-1), dxc)
                fIm[ik,:] = -ddx(beta*flx_k[ik,:], np.roll(beta*flx_k[ik,:],-1), dxc)   
                flx_contribution_from_stage_k[ik,:] = AEx[-1,ik]*(1 - beta)*flx_k[ik,:] + AIm[-1,ik]*beta*flx_k[ik,:]
                flx_HO += flx_contribution_from_stage_k[ik,:]  
            if iterFCT:
                previous = np.full(nx, False) #[True if cf[i] <= 1. else False for i in range(nx)] # [i] at i-1/2 # determines whether FCT also uses field[it] for bounds (True/False) # could use further consideration
                field[it+1] = lim.iterFCT(flx_HO, dxc, dt, uf, cf, beta, field[it], previous=previous, niter=nIter) # also option for ymin and ymax         
            else:     
                field[it+1] = field_k.copy()
            if output_substages: 
                plt.title('Substage fields during time step ' + str(it+1))
                plt.legend()
                plt.show()            
                if iterFCT: 
                    print('WARNING: substage output will be inconsistent with final field due to limiting.')
                    logging.info('WARNING: substage output will be inconsistent with final field due to limiting.')  

    return field


def set_offcentring(nx, blend, u_setting, c, clim):
    """Function to set the off-centring in time beta based on a given blend."""
    beta = np.ones(nx) # [i] at i-1/2

    if u_setting == 'varying_space_time' and blend != 'sm':
        raise ValueError('For a varying velocity field, off-centering in time is only implemented for a smooth transition from 0 to 1 with 1-1/c.')
    else:
        if blend == 'off':    
            for i in range(nx):
                if c[i] <= clim: 
                    beta[i] = 0.
        elif blend == 'rExlIm':
            for i in range(int(nx/2)):
                beta[i] = 0. 
        elif blend == 'rExlIm_sm': # linear smoothing for 1/10 of the domain
            for i in range(int(nx/10)):
                beta[i] = (int(nx/10) - i)/int(nx/10)
            for i in range(int(nx/10), int(nx/2)):
                beta[i] = 0. 
            for i in range(int(nx/2), int(nx/2) + int(nx/10)):
                beta[i] = (i - int(nx/2))/int(nx/10)
        elif blend == 'sm': # smooth transition from 0 to 1 with 1-1/c
            for i in range(nx):
                beta[i] = np.maximum(0., 1. - 1./c[i])
        elif blend == 'sm_centredspace': # smooth transition from 0 to 1 with 1-1/c and then spatial smoothing between neighbouring points. Centered in space: 0.25*[i-3/2] + 0.5*[i-1/2] + 0.25*[i+1/2] for face i-1/2
            for i in range(nx):
                beta[i] = np.maximum(0., 1. - 1./c[i])
            beta = 0.25*np.roll(beta, 1) + 0.5*beta + 0.25*np.roll(beta, -1) # centered in space: 0.25*[i-3/2] + 0.5*[i-1/2] + 0.25*[i+1/2] for face i-1/2
        elif blend == 'Im':
            beta = np.ones(nx)
        elif blend == 'Ex':
            beta = np.zeros(nx)
        else:
            raise ValueError('Blend in off-centering not recognised.')
        
    return beta