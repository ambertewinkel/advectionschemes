# Module file with various analytic function to initialise and 
# compare with FD solutions in main.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np

def analytic1(x, nt=0., c=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has values from the function 
    y = 0.5*(1-cos(2pi(x-a)/(b-a))), in the range of the domain enclosed by a and b. 
    Outside of this region, the array elements are zero.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    nt  : integer, number of time steps advected
    c   : float or 1D array of floats, Courant number for advection (c = u*dt/dx)
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    a, b = 0.1, 0.5
    psi = np.zeros(len(x))
    dx = x[1] - x[0]
    xmax = x[-1] + dx       # size of domain (assuming periodicity)
    x0 = (x - c*nt*dx)%xmax # initial x that input x corresponds to after advection (u*t = c*nt*dx)
    for i in range(len(x)):
        if x0[i] >= a and x0[i] < b: # define nonzero region
            psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a)/(b-a)))

    return psi

def analytic2(x, nt=0., c=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has output values 1 in the range
    of the domain enclosed by a and b and outside of this region, 0. This emulates a 
    step function. 
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the step 
        function for
    nt  : integer, number of time steps advected (nt = t/dt)
    c   : float or 1D array of floats, Courant number for advection (c = u*dt/dx)
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    a, b = 0.1, 0.5
    psi = np.zeros(len(x))
    dx = x[1] - x[0]
    xmax = x[-1] + dx       # size of domain (assuming periodicity)
    x0 = (x - c*nt*dx)%xmax # initial x that input x corresponds to after advection (u*t = c*nt*dx)
    for i in range(len(x)):
        if x0[i] >= a + 1.E-6 and x0[i] < b - 1.E-6: # define nonzero region
            psi[i] = 1.

    return psi