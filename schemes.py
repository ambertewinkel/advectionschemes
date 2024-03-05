# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, CTCS, Upwind, MPDATA, CNBS, and CNCS, and then BTBS, BTFS, BTCS with direct and iterative solvers. Lastly, hybrid scheme with BTBS + 1 Jacobi iteration for implicit and MPDATA for explicit
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np
import utils as ut
import solvers as sv

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init.copy()

    # Time stepping
    for it in range(nt):
        field = field - dt*(np.roll(uf,-1)*field - uf*np.roll(field,1))/dxc
 
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init.copy()

    # Time stepping
    for it in range(nt):
        field = field - dt*(np.roll(uf*field,-1) - uf*field)/dxc

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init. Defined at centers
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init.copy()

    # First time step is forward in time, centered in space (FTCS)
    field = field_old - 0.5*dt*(np.roll(uf,-1)*(field - np.roll(field,-1)) - uf*(np.roll(field,1) + field))/dxc

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - dt*(np.roll(uf,-1)*(field - np.roll(field,-1)) - uf*(np.roll(field,1) + field))/dxc
        field_old = field.copy()
        field = field_new.copy()
        
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init.copy()
    field_new = np.zeros(len(init))
    
    cc = 0.5*dt*(uf + np.roll(uf,-1))/dxc # sum for cc[i] is over the faces at i-1/2 and i+1/2

    # Time stepping
    for it in range(nt):
        for i in range(len(init)):
            if cc[i] >= 0.0:  # FTBS when u >= 0, FTFS when u < 0
                field_new[i] = field[i] - dt*(np.roll(uf,-1)[i]*field[i] - uf[i]*np.roll(field,1)[i])/dxc[i]
            else: 
                field_new[i] = field[i] - dt*(np.roll(uf*field,-1)[i] - uf[i]*field[i])/dxc[i]
        field = field_new.copy()    
    
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = np.linalg.solve(M, field)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.GaussSeidel(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.SymmetricGaussSeidel(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = np.linalg.solve(M, field)
    
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.BackwardGaussSeidel(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.SymmetricGaussSeidel(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = np.linalg.solve(M, field)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i] - dt*uf[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]
        M[i, (i+1)%len(init)] = dt*np.roll(uf,-1)[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.GaussSeidel(M, field, field, niter)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + 0.5*dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -0.5*dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        rhs = (1 - 0.5*dt*np.roll(uf,-1)/dxc)*field + 0.5*dt*uf*np.roll(field,1)/dxc
        field = np.linalg.solve(M, rhs)

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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
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
        rhs = (1 - 0.25*dt*(np.roll(uf,-1) - uf)/dxc)*field + 0.25*dt*uf*np.roll(field,1)/dxc - 0.25*dt*np.roll(uf,-1)*np.roll(field,-1)/dxc
        field = np.linalg.solve(M, rhs)

    return field

def MPDATA(init, nt, dt, uf, dxc, eps=1e-6):
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Initialisation
    field = np.zeros(len(init))
    field_old = init
    field_FP = np.zeros(len(init))
    A = np.zeros(len(init)) # A[i] is A_{i-1/2}
    V = np.zeros(len(init)) # Same index shift as for A

    U = uf*dt/np.roll(dxc,1) # U[i] is defined at i-1/2 # !!! keep dxc separate form U -- keep the code close to the fundamental equations

    # Time stepping
    for it in range(nt):
        # First pass  

        field_FP = field_old - flux(field_old, np.roll(field_old,-1), np.roll(U,-1)) + flux(np.roll(field_old,1), field_old, U)

        # Second pass
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps)
        V = (abs(U) - U*U)*A
        #fluxtemp = flux()
        field = field_FP - flux(field_FP, np.roll(field_FP,-1), np.roll(V,-1)) + flux(np.roll(field_FP,1), field_FP, V)
        field_old = field

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

def hybrid_MPDATA_BTBS1J(init, nt, dt, uf, dxc, eps=1e-6):
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
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Initialisation
    field = np.zeros(len(init))
    field_old = init
    field_FP = np.zeros(len(init))
    A = np.zeros(len(init)) # A[i] is A_{i-1/2}
    V = np.zeros(len(init)) # Same index shift as for A
    U = uf*dt/np.roll(dxc,1) # U[i] is defined at i-1/2

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc # assumes uf is positive when pointed to the right (i.e., direction of increasing x)
    beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is defined at i-1/2 # 0: explicit, 1: implicit  

    # Time stepping
    for it in range(nt):
        for i in range(len(cc)):
            if beta[i] == False: # MPDATA
                # First pass  
                field_FP[i] = field_old[i] - flux(field_old[i], np.roll(field_old,-1)[i], np.roll(U,-1)[i]) + flux(np.roll(field_old,1)[i], field_old[i], U[i])

                # Second pass
                A[i] = (field_FP[i] - np.roll(field_FP,1)[i])/(field_FP[i] + np.roll(field_FP,1)[i] + eps)
                V[i] = (abs(U[i]) - U[i]*U[i])*A[i]
                field[i] = field_FP[i] - flux(field_FP[i], np.roll(field_FP,-1)[i], np.roll(V,-1)[i]) + flux(np.roll(field_FP,1)[i], field_FP[i], V[i])
        field_temp = field.copy()
        for i in range(len(cc)):
            if beta[i] == True: # BTBS with 1 Jacobi iteration 
                field[i] = (field_temp[i] + dt*uf[i]*np.roll(field_temp,1)[i]/dxc[i])/(1 + np.roll(uf,-1)[i]*dt/dxc[i])
        field_old = field

    return field

"""
# MPDATA
    # Time stepping
    for it in range(nt):
        # First pass  
        field_FP = field_old - flux(field_old, np.roll(field_old,-1), np.roll(U,-1)) + flux(np.roll(field_old,1), field_old, U)

        # Second pass
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps)
        V = (abs(U) - U*U)*A
        field = field_FP - flux(field_FP, np.roll(field_FP,-1), np.roll(V,-1)) + flux(np.roll(field_FP,1), field_FP, V)
        field_old = field

    return field


# BTBS+Jacobi
        # Define initial condition
    field = init.copy()

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + dt*np.roll(uf,-1)[i]/dxc[i]
        M[i, i-1] = -dt*uf[i]/dxc[i]

    # Time stepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

    return field
"""

def hybrid_Upwind_BTBS1J(init, nt, dt, uf, dxc):
    """
    This functions implements 
    Explicit: upwind scheme (assuming a 
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
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Initialisation
    field = np.zeros(len(init))
    field_old = init.copy()
    flx = np.zeros(len(init))

    # Criterion explicit/implicit
    cc = 0.5*dt*(np.roll(uf,-1) + uf)/dxc # assumes uf is positive when pointed to the right (i.e., direction of increasing x)
    beta = np.invert((np.roll(cc,1) <= 1.)*(cc <= 1)) # beta[i] is defined at i-1/2 # 0: explicit, 1: implicit  
    #beta = np.maximum.reduce([np.zeros(len(cc)), 1 - 1/cc, 1 - 1/np.roll(cc,1)]) # beta[i] is defined at i-1/2 # 0: fully explicit, 1: fully implicit 
    
    ufp = 0.5*(uf + abs(uf)) # uf[i] is defined at i-1/2
    ufm = 0.5*(uf - abs(uf))

    # Time stepping
    for it in range(nt):
        flx = ufp*np.roll(field_old,1) + ufm*field_old # flx[i] is defined at i-1/2
        rhs = field_old - dt*(np.roll((1. - beta)*flx,-1) - (1. - beta)*flx)/dxc
        for i in range(len(cc)):
            if beta[i] != 0.0 or np.roll(beta,-1)[i] != 0.0:
                field[i] = (rhs[i] + dt*uf[i]*np.roll(field_old,1)[i]/dxc[i])/(1 + np.roll(uf,-1)[i]*dt/dxc[i])
            else:
                field[i] = rhs[i]
        field_old = field.copy()
    return field

def hybrid_Upwind_Upwind1J(init, nt, dt, uf, dxc, eps=1e-6):
    print()