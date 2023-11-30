# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, FTCS, CTBS, CTFS, CTCS, MPDATA

import numpy as np

def ftbs(init, nt, c):
    """
    This function computes the FTBS (forward in time, backward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Time stepping
    for it in range(1, nt):
        field = field - c*(field - np.roll(field,1))

    return field

def ftfs(init, nt, c):
    """
    This function computes the FTFS (forward in time, forward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Time stepping
    for it in range(1, nt):
        field = field - c*(np.roll(field,-1) - field)

    return field

def ftcs(init, nt, c):
    """
    This function computes the FTCS (forward in time, centered in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = init

    # Time stepping
    for it in range(1, nt):
        field = field - 0.5*c*(np.roll(field,-1) - np.roll(field,1))

    return field

def ctbs(init, nt, c):
    """
    This function computes the CTBS (centered in time, backward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # First time step is forward in time, backward in space (FTBS)
    field = field_old - c*(field_old - np.roll(field_old,1))

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - 2*c*(field - np.roll(field,1))
        field_old = field
        field = field_new
        
    return field

def ctfs(init, nt, c):
    """
    This function computes the CTFS (centered in time, forward in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # First time step is forward in time, forward in space (FTFS)
    field = field_old - c*(np.roll(field_old,-1) - field_old)

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - 2*c*(np.roll(field,-1) - field)
        field_old = field
        field = field_new
        
    return field

def ctcs(init, nt, c):
    """
    This function computes the CTCS (centered in time, centered in space)
    finite difference scheme for an initial field, number of time steps nt
    with length dt, and a given Courant number. A periodic spatial domain 
    is assumed.
    --- Input ---
    init    : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: length of init.
    """

    # Setup and initial condition
    field = np.zeros(len(init))
    field_old = init

    # First time step is forward in time, centered in space (FTCS)
    field = field_old - 0.5*c*(np.roll(field_old,-1) - np.roll(field_old,1))

    # Time stepping
    for it in range(1, nt):
        field_new = field_old - c*(np.roll(field,-1) - np.roll(field,1))
        field_old = field
        field = field_new
        
    return field

def upwind():
    print()

def artdiff():
    print()

def semiLag():
    print()

def btbs(): #implicit
    print()

def btcs(): #implicit
    print()

def cnbs(): #implicit
    print()

def cncs(): #implicit
    print()

def mpdata(init, nt, c, eps=1e-6):
    """
    This functions implements the MPDATA scheme without a gauge, assuming a 
    constant velocity, i.e., a single local Courant number input, and a 
    periodic spatial domain.
    Reference (1): P. Smolarkiewicz and L. Margolin. MPDATA: A finite-difference 
    solver for geophysical flows. J. Comput. Phys., 140:459-480, 1998.
    --- Input ---
    init : array of floats, initial field to advect
    nt      : integer, total number of time steps to take
    c       : Courant number. c = u*dt/dx where u is the velocity and dx 
            is the spatial discretisation
    eps     : float, optional. Small number to avoid division by zero.
    --- Output --- 
    field   : 1D array of floats. Outputs the final timestep after advecting 
            the initial condition. Dimensions: (nt+1) x length of init.
    """

    # Initialisation
    field = np.zeros(len(init))
    field_old = init
    field_FP = np.zeros(len(init))
    A = np.zeros(len(init)) # shift index by half as compared to Ref (1)
    V = np.zeros(len(init)) # pseudo-velocity. Same index shift as for A

    # Timestepping
    for it in range(nt):
        # First pass - any U is here c as velocity is constant    
        field_FP = field_old - flux(field_old, np.roll(field_old,-1), c) + flux(np.roll(field_old,1), field_old, c)

        # Second pass
        A = (np.roll(field_FP,-1) - field_FP)/(np.roll(field_FP,-1) + field_FP + eps)
        V = (abs(c) - c*c)*A
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
