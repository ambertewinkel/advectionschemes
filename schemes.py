# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, MPDATA

import numpy as np
import utils as ut
import solvers as sv

def FTBS(init, nt, c):
    """
    This function computes the FTBS (forward in time, backward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # Time stepping
    for it in range(nt):
        field = field - c_arr*(field - np.roll(field,1))

    return field

def FTFS(init, nt, c):
    """
    This function computes the FTFS (forward in time, forward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # Time stepping
    for it in range(nt):
        field = field - c_arr*(np.roll(field,-1) - field)

    return field

def FTCS(init, nt, c):
    """
    This function computes the FTCS (forward in time, centered in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # Time stepping
    for it in range(nt):
        field = field - 0.5*c_arr*(np.roll(field,-1) - np.roll(field,1))

    return field

def CTBS(init, nt, c):
    """
    This function computes the CTBS (centered in time, backward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # First time step is forward in time, backward in space (FTBS)
    field = field_old - c_arr*(field_old - np.roll(field_old,1))

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - 2*c_arr*(field - np.roll(field,1))
        field_old = field
        field = field_new
        
    return field

def CTFS(init, nt, c):
    """
    This function computes the CTFS (centered in time, forward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # First time step is forward in time, forward in space (FTFS)
    field = field_old - c_arr*(np.roll(field_old,-1) - field_old)

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - 2*c_arr*(np.roll(field,-1) - field)
        field_old = field
        field = field_new
        
    return field

def CTCS(init, nt, c):
    """
    This function computes the CTCS (centered in time, centered in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # First time step is forward in time, centered in space (FTCS)
    field = field_old - 0.5*c_arr*(np.roll(field_old,-1) - np.roll(field_old,1))

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - c_arr*(np.roll(field,-1) - np.roll(field,1))
        field_old = field
        field = field_new
        
    return field

def Upwind(init, nt, c): # FTBS when u >= 0, FTFS when u < 0
    """
    This function computes the upwind (FTBS when u>=0, FTFS when u<0)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed. dt and dx are assumed to be positive.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation 
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init
    field_new = init.copy()
    
    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init)) # !!! to do: make the wind go along perhaps somehow with the field?

    # Time stepping
    for it in range(nt):
        for i in range(len(init)):
            if c_arr[i] >= 0.0:
                field_new[i] = field[i] - c_arr[i]*(field[i] - np.roll(field,1)[i])
            else: 
                field_new[i] = field[i] - c_arr[i]*(np.roll(field,-1)[i] - field[i])
        field = field_new.copy()    
    
    return field

def ArtDiff():
    print()

def SemiLag():
    print()

def BTBS(init, nt, c):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + c[i] # assume c is @i and not @i-1 and doesn't change over time
        M[i, i-1] = -c[i]

    # Timestepping
    for it in range(nt):
        field = np.linalg.solve(M, field)

    return field

def BTBS_Jacobi(init, nt, c, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + c[i] # assume c is @i and not @i-1 and doesn't change over time
        M[i, i-1] = -c[i]

    # Timestepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

    return field

def BTBS_GaussSeidel(init, nt, c, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, backward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 + c[i] # assume c is @i and not @i-1 and doesn't change over time
        M[i, i-1] = -c[i]

    # Timestepping
    for it in range(nt):
        field = sv.GaussSeidel(M, field, field, niter)

    return field

def BTFS(init, nt, c):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - c[i] # assume c is @i and not @i+1 and doesn't change over time
        M[i, (i+1)%len(init)] = c[i]

    # Timestepping
    for it in range(nt):
        field = np.linalg.solve(M, field)
    
    return field

def BTFS_Jacobi(init, nt, c, niter=1):
    """
    This functions implements the BTFS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - c[i] # assume c is @i and not @i+1 and doesn't change over time
        M[i, (i+1)%len(init)] = c[i]

    # Timestepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

    return field

def BTFS_GaussSeidel(init, nt, c, niter=1):
    """
    This functions implements the BTBS scheme (backward in time, forward in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1 - c[i] # assume c is @i and not @i+1 and doesn't change over time
        M[i, (i+1)%len(init)] = c[i]

    # Timestepping
    for it in range(nt):
        field = sv.BackwardGaussSeidel(M, field, field, niter)

    return field


def BTCS(init, nt, c):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1  
        M[i, i-1] = -0.5*c[i] # assume c is @i and doesn't change over time
        M[i, (i+1)%len(init)] = 0.5*c[i]

    # Timestepping
    for it in range(nt):
        field = np.linalg.solve(M, field)

    return field

def BTCS_Jacobi(init, nt, c, niter=1):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Jacobi iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1  
        M[i, i-1] = -0.5*c[i] # assume c is @i and doesn't change over time
        M[i, (i+1)%len(init)] = 0.5*c[i]

    # Timestepping
    for it in range(nt):
        field = sv.Jacobi(M, field, field, niter)

    return field

def BTCS_GaussSeidel(init, nt, c, niter=1):
    """
    This functions implements the BTCS scheme (backward in time, centered in 
    space, implicit), assuming a constant velocity (input through the Courant 
    number) and a periodic spatial domain. It uses the Gauss-Seidel iteration method.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    niter   : number of iterations used for the Jacobi iterative method, default=1
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """
    # Define initial condition
    field = init

    # Define the matrix to solve
    M = np.zeros((len(init), len(init)))
    for i in range(len(init)): 
        M[i,i] = 1  
        M[i, i-1] = -0.5*c[i] # assume c is @i and doesn't change over time
        M[i, (i+1)%len(init)] = 0.5*c[i]

    # Timestepping
    for it in range(nt):
        field = sv.GaussSeidel(M, field, field, niter)

    return field

def CNBS(): #implicit
    print()

def CNCS(): #implicit
    print()

def MPDATA(init, nt, c, eps=1e-6):
    """
    This functions implements the MPDATA scheme without a gauge, assuming a 
    constant velocity (input through the Courant number) and a 
    periodic spatial domain.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : float or array of floats. Courant number. c = u*dt/dx where u 
            is the velocity, dt the timestep, and dx the spatial discretisation
    eps     : float, optional. Small number to avoid division by zero.
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Initialisation
    field = np.zeros(len(init))
    field_old = init
    field_FP = np.zeros(len(init))
    A = np.zeros(len(init)) # shift index by half as compared to Ref (1)
    V = np.zeros(len(init)) # pseudo-velocity. Same index shift as for A

    # Ensure c has the right dims for it loop
    c_arr = ut.to_vector(c, len(init))

    # Timestepping
    for it in range(nt):
        # First pass  
        field_FP = field_old - flux(field_old, np.roll(field_old,-1), c_arr) + flux(np.roll(field_old,1), field_old, c_arr)

        # Second pass
        A = (np.roll(field_FP,-1) - field_FP)/(np.roll(field_FP,-1) + field_FP + eps)
        V = (abs(c_arr) - c_arr*c_arr)*A
        field = field_FP - flux(field_FP, np.roll(field_FP,-1), V) + flux(np.roll(field_FP,1), field_FP, np.roll(V,1))
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