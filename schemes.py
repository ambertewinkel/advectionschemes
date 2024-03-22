# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, CTCS, Upwind, MPDATA, CNBS, and CNCS, and then BTBS, BTFS, BTCS with direct and iterative solvers. Lastly, hybrid scheme with BTBS + 1 Jacobi iteration for implicit and MPDATA for explicit
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import utils as ut
import solvers as sv
import matplotlib.pyplot as plt

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
        spatial = np.where(cc >= 0.0, np.roll(uf,-1)*field[it] - uf*np.roll(field[it],1), np.roll(uf*field[it],-1) - uf*field[it]) # BS when u >= 0, FS when u < 0
        field[it+1] = field[it] - dt*spatial/dxc
    
    return field

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

def CNBS(init, nt, dt, uf, dxc): # Crank-Nicolson (implicit)
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

def CNCS(init, nt, dt, uf, dxc): # Crank-Nicolson (implicit)
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

def MPDATA(init, nt, dt, uf, dxc, dxf, eps=1e-6):
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
    --- Output --- 
    field   : 2D array of floats. Outputs each timestep of the field while advecting 
            the initial condition. Dimensions: nt+1 x length of init
    """

    # Initialisation
    field = np.zeros((nt+1, len(init)))
    field[0] = init.copy()

    # Time stepping
    for it in range(nt):
        # First pass  
        flx_FP = flux(np.roll(field[it],1), field[it], uf) # flx_FP[i] is at i-1/2
        field_FP = field[it] - dt*(np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, np.roll(uf,1)/abs(np.roll(uf,1)))
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps) # A[i] is at i-1/2
        V = A*np.roll(uf,1)/(0.5*np.roll(dxf,-1))*(dx_up - 0.5*dt*uf) # Same index shift as for A
        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP - dt*(np.roll(flx_SP,-1) - flx_SP)/dxc

    return field

def hybrid_MPDATA_BTBS1J(init, nt, dt, uf, dxc, dxf, eps=1e-6, do_beta='switch'):
    """
    This functions implements 
    Explicit: MPDATA scheme (without a gauge, assuming a 
    constant velocity and a 
    periodic spatial domain)
    Implicit: BTBS with 1 Jacobi iteration
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
    field_FP = np.zeros(len(init))

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    if do_beta == 'switch':
        beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit 
    elif do_beta == 'blend':
        beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 
    else:
        print('Error: do_beta must be either "switch" or "blend"')

    xi = np.maximum(1 - 2*beta, np.zeros(len(cc))) # xi[i] is at i-1/2

    # Time stepping
    for it in range(nt):
        # First pass
        flx_FP = flux(np.roll(field[it],1), field[it], uf) # flx_FP[i] is at i-1/2 # upwind
        rhs = field[it] - dt*(np.roll((1. - beta)*flx_FP,-1) - (1. - beta)*flx_FP)/dxc
        for i in range(len(cc)):
            if beta[i] != 0.0 or np.roll(beta,-1)[i] != 0.0: # BTBS1J
                aii = 1 + np.roll(beta*uf,-1)[i]*dt/dxc[i]
                aiim1 = -dt*beta[i]*uf[i]/dxc[i]
                field_FP[i] = (rhs[i] - aiim1*np.roll(field[it],1)[i])/aii            
            else:
                field_FP[i] = rhs[i]
        field = field_FP.copy()

        # Second pass
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, np.roll(uf,1)/abs(np.roll(uf,1)))
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps) # A[i] is at i-1/2
        V = A*np.roll(uf,1)/(0.5*np.roll(dxf,-1))*(dx_up - 0.5*dt*xi*uf) # Same index shift as for A
        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP + dt*(-np.roll(flx_SP,-1) + flx_SP)/dxc                

    return field

def hybrid_Upwind_BTBS1J(init, nt, dt, uf, dxc):
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

    # Criterion explicit/implicit # !!! include do_beta criterion
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit  
    #beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 

    # Time stepping
    for it in range(nt):
        flx = flux(np.roll(field[it],1), field[it], uf) # flx[i] is at i-1/2 # upwind
        rhs = field[it] - dt*(np.roll((1. - beta)*flx,-1) - (1. - beta)*flx)/dxc
        # for ... # include number of iterations here!!!
        for i in range(len(cc)):
            if beta[i] != 0.0 or np.roll(beta,-1)[i] != 0.0: # BTBS1J
                aii = 1 + np.roll(beta*uf,-1)[i]*dt/dxc[i]
                aiim1 = -dt*beta[i]*uf[i]/dxc[i]
                field[it+1,i] = (rhs[i] - aiim1*np.roll(field[it],1)[i])/aii
            else:
                field[it+1,i] = rhs[i]

    return field

def hybrid_Upwind_Upwind1J(init, nt, dt, uf, dxc):
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

    # Criterion explicit/implicit # !!! include do_beta criterion
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc
    beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is at i-1/2 # 0: explicit, 1: implicit  
    #beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is at i-1/2 # 0: fully explicit, 1: fully implicit 

    ufp = 0.5*(uf + abs(uf)) # uf[i] is at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Time stepping
    for it in range(nt):
        flx = flux(np.roll(field[it],1), field[it], uf) # flx[i] is at i-1/2
        rhs = field[it] - dt*(np.roll((1. - beta)*flx,-1) - (1. - beta)*flx)/dxc
        for i in range(len(cc)):
            if beta[i] != 0.0 or np.roll(beta,-1)[i] != 0.0:
                aii = 1. + dt*(np.roll(beta*ufp,-1)[i] - beta[i]*ufm[i])/dxc[i]
                aiim1 = -dt*beta[i]*ufp[i]/dxc[i]
                aiip1 = dt*np.roll(beta*ufm,-1)[i]/dxc[i]
                field[it+1,i] = (rhs[i] - aiim1*np.roll(field[it],1)[i] - aiip1*np.roll(field[it],-1)[i])/aii
            else:
                field[it+1,i] = rhs[i]

    return field

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