"""
Standalone code to run RK2QC separately, without all the machinery of the main code.
Date: 03-12-2024
Author: Amber te Winkel
Email: a.j.tewinkel@pgr.reading.ac.uk
"""


import numpy as np
import matplotlib.pyplot as plt


def RK2QC(init, nt, dt, uf, dxc, kmax=2):
    """This scheme solves second-order Runge-Kutta quasi-cubic scheme. See HW notes sent on 27-11-2024.
    Assumes uniform grid and uf>0.
    --- in ---
    init : 1D array of floats, initial condition
    nt   : integer, number of time steps
    dt   : float, time step
    uf   : 1D array of floats, velocity
    dxc  : 1D array of floats, cell size
    kmax : integer, number of iterations
    --- out ---
    field : 2D array of floats, field at each time step
    """

    nx = len(init)
    field = np.zeros((nt+1, nx))
    field[0] = init.copy()
    fieldh_HO_n, flx_HO_n, fieldh_1st_km1, flx_1st_km1, fieldh_HOC_km1, flx_HOC_km1, rhs, beta = np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx), np.zeros(nx)
    M = np.zeros((nx, nx))

    c = dt*uf/dxc # assumes uniform grid
    alpha = np.maximum(0.5, 1. - 1./c) # assumes uniform grid # alpha[i] is at i-1/2
    for i in range(nx):
        beta[i] = 0. if c[i] < 0.8 else 1 # beta[i] is at i-1/2

    for i in range(nx):	# includes flx_1st_k # based on implicit upwind matrices above.
        M[i,i] = 1. + dt*np.roll(alpha*beta*uf,-1)[i]/dxc[i]
        M[i,i-1] = -dt*alpha[i]*beta[i]*uf[i]/dxc[i]

    for it in range(nt):
        field[it+1] = field[it].copy() # not actually in the equations, this is to make the computer code more concise
        for k in range(kmax):
            for i in range(nx):
                fieldh_HO_n[i] = quadh(field[it,i-2], field[it,i-1], field[it,i]) # [i] defined at i-1/2
                flx_HO_n[i] = (1. - alpha[i])*uf[i]*fieldh_HO_n[i] # [i] defined at i-1/2
                fieldh_1st_km1[i] = field[it+1,i-1] # upwind # [i] defined at i-1/2 # not actually field[it+1], this is to make the computer code more concise
                flx_1st_km1[i] = alpha[i]*(1. - beta[i])*uf[i]*fieldh_1st_km1[i] # [i] defined at i-1/2
                fieldh_HOC_km1[i] = fieldh_HO_n[i] - fieldh_1st_km1[i] # [i] defined at i-1/2
                flx_HOC_km1[i] = alpha[i]*uf[i]*fieldh_HOC_km1[i] # [i] defined at i-1/2
            for i in range(nx):
                rhs[i] = field[it,i] - dt*(ddx(flx_HO_n[i], flx_HO_n[(i+1)%nx], dxc[i]) + \
                        ddx(flx_1st_km1[i], flx_1st_km1[(i+1)%nx], dxc[i]) + \
                        ddx(flx_HOC_km1[i], flx_HOC_km1[(i+1)%nx], dxc[i])) # [i] defined at i
            field[it+1] = np.linalg.solve(M, rhs)    

    return field


def quadh(fm1, f, fp1):
    """This quadratic interpolation for f[i+1/2] leads to a cubic approximation when put in the FV ddx scheme. The quadratic interpolation matches the integral of the polynomial to the integral of the field over the three cells. See notes sent by James Kent on 28-11-2024.
    --- in ---
    fm1 : f[i-1]
    f   : f[i]
    fp1 : f[i+1]
    --- out --- 
    f[i+1/2] 
    """
    return (2.*fp1 + 5.*f - fm1)/6.


def ddx(fmh, fph, dxc):
    """This function computes the first derivative of a field f with respect to x with a finite-volume method. 
    dfdx[i] = (f_{i+1/2} - f_{i-1/2})/dx
    fmh : f_{i-1/2}
    fph : f_{i+1/2}
    """
    return (fph - fmh)/dxc


def sine(x, xmax, u=0., t=0., shifty=0.5, ampl=0.5, shiftx=0.):
    """This function returns an array from input array x advected by velocity u for a time t.
    The initial condition has values from the function y = shifty + ampl*sin(2pi(x-shiftx)/xmax).
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    shifty: float, y shift of the sine wave
    ampl : float, amplitude of the sine wave
    shiftx: float, x shift of the sine wave
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    psi = shifty + ampl*np.sin(2*np.pi*(x0-shiftx)/xmax)

    return psi


def combi(x, xmax, u=0., t=0., shift=0., ampl=1., a=0., b=0.5, c=0.6, d=0.8):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has output values 1 in the range
    of the domain enclosed by a and b and outside of this region, 0. This function is 
    a combination of the cosinebell and tophat functions.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    shift: float, shift of the cosine bell and top hat
    ampl : float, amplitude of the cosine bell and top hat
    a   : float, left boundary of cosine bell
    b   : float, right boundary of cosine bell
    c   : float, left boundary of top hat
    d   : float, right boundary of top hat
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    
    # Define nonzero region of the cosine bell
    if a < b:
        psi = shift + ampl*np.where((x0 >= a) & (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a)/(b-a))), 0.)
    else:
        psi = shift + ampl*np.where((x0 >= a) | (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a+xmax)/(b-a+xmax))), 0.)

    # Define nonzero region of the top hat
    if c < d:
        psi = shift + ampl*np.where((x0 >= c) & (x0 <= d), 1., psi)
    else:
        psi = shift + ampl*np.where((x0 >= c) | (x0 <= d), 1., psi)

    return psi


def rmse(field, analytic, dx):
    """
    This function calculates the root-mean-square-error (RMSE) for a finite difference (FD)
    scheme as compared to the analytic solution. Assumed is a 1D spatial input for a
    certain point in time t.
    --- Input --- 
    field       : 1D array of floats, FD spatial field at t
    analytic    : analytic solution at t
    dx          : spatial discretisation
    --- Output --- 
    rmse     : 1D array of floats, root-mean-square-error.
    """
    rmse = np.zeros(len(field))
    rmse = np.sqrt((np.sum(dx*(field-analytic)*(field-analytic)))/(np.sum(dx*analytic*analytic) + 1e-16))

    return rmse


def main():
    """Main function to run standalone code based on the RK2QC scheme."""

    testfunction = combi # sine or combi
    dt = 0.01                   # time step
    nt = 100                  # number of time steps
    nx = 40                     # number of points in space
    xmax = 1.                   # physical domain parameters
    uconstant = 1.          # constant velocity (u = 1 -> C = 0.4)
    dx = np.ones(nx)*xmax/nx                # spatial resolution
    uf = np.ones(nx)*uconstant

    # Define the grid
    x = np.linspace(0., xmax, nx, endpoint=False)

    # Calculate analytic solutions
    initial = testfunction(x, xmax, t=0.)
    analytic = testfunction(x, xmax, u=uconstant, t=nt*dt)

    # Solve the advection equation with the RK2QC scheme
    field = RK2QC(initial, nt, dt, uf, dx)

    # Calculate and print the RMSE
    RMSE = rmse(field[-1], analytic, dx)
    print('RMSE:', RMSE)

    # Plot the results
    plt.plot(x, initial, label='Initial')
    plt.plot(x, analytic, label='Analytic')
    plt.plot(x, field[-1], label='Numerical')
    plt.legend()
    plt.show()

    return 0

if __name__ == '__main__':
    main()