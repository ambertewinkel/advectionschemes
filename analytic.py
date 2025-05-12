# Module file with various analytic function to initialise and 
# compare with FD solutions in main.py
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


import numpy as np


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


def sine_yshift(x, xmax, u=0., t=0., shifty=10.0, ampl=0.5, shiftx=0.):
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


def cosbell(x, xmax, u=0., t=0., shift=0., ampl=1., a=0., b=0.5):
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
    ampl : float, amplitude of the cosine bell
    a   : float, left boundary of cosine bell
    b   : float, right boundary of cosine bell
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

    return psi

def cosbell_yshift(x, xmax, u=0., t=0., shift=10., ampl=1., a=0., b=0.5):
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
    ampl : float, amplitude of the cosine bell
    a   : float, left boundary of cosine bell
    b   : float, right boundary of cosine bell
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

    return psi

def tophat(x, xmax, u=0., t=0., shift=0., ampl=1., a=0.6, b=0.8):
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
    ampl : float, amplitude of the top hat
    a   : float, left boundary of top hat
    b   : float, right boundary of top hat
    --- Output ---
    psi : 1D array of floats, result from function at the points defined in x
    """    
    psi = np.zeros(len(x))
    x0 = (x - u*t)%xmax

    # Define nonzero region of the top hat
    if a < b:
        psi = shift + ampl*np.where((x0 >= a + 1.E-6) & (x0 <= b - 1.E-6), 1., 0.)
    else:
        psi = shift + ampl*np.where((x0 >= a + 1.E-6) | (x0 <= b - 1.E-6), 1., 0.)

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


def analytic_constant(x, xmax, u=0., t=0., val=10.0):
    """Returns a constant value at default 10 for the initial condition."""
    psi = np.full(len(x), val)
    return psi


def velocity_varying_space(x, l=2.*np.pi):
    """This function returns a velocity field that is varying in space. The velocity field is given by u = 5.5 + 4.5*sin(lx), where l is a real number, default 2*pi."""
    u = 5.5 + 4.5*np.sin(l*x)
    return u

def velocity_varying_space2(x, l=2.*np.pi):
    """This function returns a velocity field that is varying in space. The velocity field is given by u = 5.5 + 4.5*sin(lx), where l is a real number, default 2*pi."""
    u = 2.5 + 1.5*np.sin(l*x)
    return u


def velocity_varying_space3(x, l=2.*np.pi):
    """This function returns a velocity field that is varying in space. The velocity field is given by u = 5.5 + 4.5*sin(lx), where l is a real number, default 2*pi."""
    u = 2.5 + 1.5*np.sin(l*(x + 0.25))
    return u