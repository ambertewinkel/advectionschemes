# Module file with various analytic function to initialise and 
# compare with FD solutions in main.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np

def cosinebell(x, xmax, u=0., t=0., shift=0., amp=1., a=0., b=0.5):
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
    shift: float, shift of the cosine bell
    amp : float, amplitude of the cosine bell
    a   : float, left boundary of cosine bell
    b   : float, right boundary of cosine bell
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax

    # Define nonzero region of the cosine bell
    if a < b:
        psi = shift + amp*np.where((x0 >= a) & (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a)/(b-a)), 0.))
    else:
        psi = shift + amp*np.where((x0 >= a) | (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a+xmax)/(b-a+xmax))), 0.)
    return psi

def tophat(x, xmax, u=0., t=0., shift=0., amp=1., a=0.6, b=0.8):
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
    shift: float, shift of the top hat
    amp : float, amplitude of the top hat
    a   : float, left boundary of top hat
    b   : float, right boundary of top hat
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax

    # Define nonzero region of the top hat
    if a < b:
        psi = shift + amp*np.where((x0 >= a + 1.E-6) & (x0 <= b - 1.E-6), 1., 0.)
    else:
        psi = shift + amp*np.where((x0 >= a + 1.E-6) | (x0 <= b - 1.E-6), 1., 0.)
    return psi

def combi(x, xmax, u=0., t=0., shift=0., amp=1., a=0., b=0.5, c=0.6, d=0.8):
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
    amp : float, amplitude of the cosine bell and top hat
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
        psi = shift + amp*np.where((x0 >= a) & (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a)/(b-a))), 0.)
    else:
        psi = shift + amp*np.where((x0 >= a) | (x0 <= b), 0.5*(1 - np.cos(2*np.pi*(x0-a+xmax)/(b-a+xmax))), 0.)

    # Define nonzero region of the top hat
    if c < d:
        psi = shift + amp*np.where((x0 >= c) & (x0 <= d), 1., psi)
    else:
        psi = shift + amp*np.where((x0 >= c) | (x0 <= d), 1., psi)
    return psi