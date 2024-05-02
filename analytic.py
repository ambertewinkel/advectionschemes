# Module file with various analytic function to initialise and 
# compare with FD solutions in main.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np

def cosinebell(x, xmax, u=0., t=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has values from the function 
    y = 0.5*(1-cos(2pi(x-a)/(b-a))), in the range of the domain enclosed by a and b. 
    Outside of this region, the array elements are zero.
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    a, b = 0.2, 0.5 # a = left boundary of wave, b = right boundary of wave
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    for i in range(len(x)):
        if a < b:
            if x0[i] >= a and x0[i] < b: # define nonzero region
                psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a)/(b-a)))
        else:
            if x0[i] >= a or x0[i] < b:
                psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a+xmax)/(b-a+xmax)))
    return psi

def tophat(x, xmax, u=0., t=0.):
    """
    This function returns an array from input array x and constants a and b advected 
    by velocity u for a time t. The initial condition has output values 1 in the range
    of the domain enclosed by a and b and outside of this region, 0. This emulates a 
    step function. 
    --- Input ---
    x   : 1D array of floats, points to calculate the result of the step 
        function for
    xmax: float, domain size
    u   : float or 1D array of floats, velocity
    t   : float, total time
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    a, b = 0.2, 0.5 # a = left boundary of wave, b = right boundary of wave
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    for i in range(len(x)):
        if a < b:
            if x0[i] >= a + 1.E-6 and x0[i] < b - 1.E-6: # define nonzero region
                psi[i] = 1.
        else:
            if x0[i] >= a + 1.E-6 or x0[i] < b - 1.E-6: # define nonzero region
                psi[i] = 1.
    return psi

def combi(x, xmax, u=0., t=0.):
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
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    a, b = 0., 0.5 # a = left boundary of cosine bell, b = right boundary of cosine bell
    c, d = 0.6, 0.8 # c = left boundary of top hat, d = right boundary of top hat
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax
    for i in range(len(x)):
        if a < b:
            if x0[i] >= a and x0[i] < b: # define nonzero region
                psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a)/(b-a)))
        else:
            if x0[i] >= a or x0[i] < b:
                psi[i] = 0.5*(1 - np.cos(2*np.pi*(x0[i]-a+xmax)/(b-a+xmax)))
        if c < d:
            if x0[i] >= c and x0[i] < d:
                psi[i] = 1.
        else:
            if x0[i] >= c or x0[i] < d:
                psi[i] = 1.
    return psi