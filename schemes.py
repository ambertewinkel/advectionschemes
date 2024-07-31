# Module file with various schemes to analyze in main.py
# Schemes included: FTBS, FTFS, CTCS, Upwind, MPDATA, CNBS, and CNCS, and then BTBS, BTFS, BTCS with direct and iterative solvers. Lastly, hybrid scheme with BTBS + 1 Jacobi iteration for implicit and MPDATA for explicit
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


import numpy as np
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
        spatial = np.where(cc >= 0., np.roll(uf,-1)*field[it] - uf*np.roll(field[it],1), np.roll(uf*field[it],-1) - uf*field[it]) # BS when u >= 0, FS when u < 0
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

    # Time stepping
    for it in range(nt):
        # First pass  
        flx_FP = flux(np.roll(field[it],1), field[it], uf) # flx_FP[i] is at i-1/2
        field_FP = field[it] - dt*(np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        A = (field_FP - np.roll(field_FP,1))/(field_FP + np.roll(field_FP,1) + eps) # A[i] is at i-1/2
        V = A*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf) # Same index shift as for A
        
        if do_limit == True: # Limit V
            corrCLimit = limit*uf
            V = np.maximum(np.minimum(V, corrCLimit), -corrCLimit)  
        
        # Smooth V
        for ismooth in range(nSmooth):
            V = 0.5*V + 0.25*(np.roll(V,1) + np.roll(V,-1))

        flx_SP = flux(np.roll(field_FP,1), field_FP, V)
        field[it+1] = field_FP - dt*(np.roll(flx_SP,-1) - flx_SP)/dxc

    return field


def MPDATA_gauge(init, nt, dt, uf, dxc):
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

    # Time stepping
    for it in range(nt):
        # First pass  
        flx_FP = flux(np.roll(field[it],1), field[it], uf) # flx_FP[i] is at i-1/2
        field_FP = field[it] - dt*(np.roll(flx_FP,-1) - flx_FP)/dxc

        # Second pass
        # Infinite gauge: multiply the pseudovelocity by 0.5 and do not divide by (field_FP + np.roll(field_FP,1) + eps), and set the first two arguments in flux() to 1.
        dx_up = 0.5*flux(np.roll(dxc,1), dxc, uf/abs(uf))
        V = 0.5*(field_FP - np.roll(field_FP,1))*uf/(0.5*dxf)*(dx_up - 0.5*dt*uf)   # V[i] is at i-1/2
        flx_SP = flux(1., 1., V)
        field[it+1] = field_FP - dt*(np.roll(flx_SP,-1) - flx_SP)/dxc

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


def aiMPDATA_gauge(init, nt, dt, uf, dxc, do_beta='switch', solver='NumPy', niter=0, do_limit=False, limit=0.5, nSmooth=0, third_order=False):
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
        # Use the first-pass field for A. A[i] is at i-1/2
        ###A = (field_FP - np.roll(field_FP,1))/2.
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

def LW3rd(init, nt, dt, uf, dxc): # Only explicit and uniform grid and velocity # previously called thirdorderinfgaugeLWMPDATA
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

    for it in range(nt):
        field_FP = field[it] - c*(field[it] - np.roll(field[it],1))
        
        #field_SP = field_FP -0.5*c*(1-c)*(np.roll(field_FP, -1) - 2*field_FP + np.roll(field_FP, 1))
        field_SP = field_FP -0.5*c*(1-c)*(np.roll(field[it], -1) - 2*field[it] + np.roll(field[it], 1))

        #field[it+1] = field_SP + c*(1-c*c)/6*(np.roll(field_SP, -1) - 3*field_SP + 3*np.roll(field_SP, 1) - np.roll(field_SP, 2))
        field[it+1] = field_SP + c*(1-c*c)/6*(np.roll(field[it], -1) - 3*field[it] + 3*np.roll(field[it], 1) - np.roll(field[it], 2))

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