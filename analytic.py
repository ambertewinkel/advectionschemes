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


def cotan(x):
    return -np.tan(x + np.pi/2)


#def analytic_velocity_varying_space(x, xmax, u, t=0., l=2.*np.pi): # !!! look into this derivation and include the l value
#    """This is the analytic solution to the advection equation with a variable velocity field in space, i.e., u=2+sin(lx), where l is real. Here we have assumed l=2*pi."""
#    psi = np.zeros(len(x))
#    psi = (- np.sin(t)*np.sin(cotan(0.5*x + 0.25*np.pi)) + np.cos(t)*np.cos(cotan(0.5*x + 0.25*np.pi)))/(np.sin(x) + 1.) # !!! change 1+sinlx to 2 + sinlx and l from 1 to 2*pi
#    return psi

def analytic_velocity_varying_space(x, xmax, u, t=0., l=2.*np.pi): # make sure l matches the velocity function below
    """This is the analytic solution to the advection equation with a variable velocity field in space, i.e., u=2+sin(lx), where l is real. Here we have assumed l=2*pi."""
    k = 1.
    psi = np.zeros(len(x))
    intdx = 2.*np.arctan((2.*np.tan(l*x/2.) + 1.)/np.sqrt(3))
    psi = 1./velocity_varying_space(x, l=l)*np.exp(k*t - intdx)
    return psi




#####def analytic_velocity_varying_space_time(x, xmax, u='varying_space_time', t=0., l=2.*np.pi, w=2.*np.pi): # !!! how to sensibly set up an analytic solution for this? - I just need to it to be exact after a full period of the temporal wave - I think that is the only thing that I can actually achieve. # !!! make sure that the analytic solution is correct after a full revolution in time (and space?).
#####    """This is the analytic solution to the advection equation with a variable velocity field in space, i.e., u=2+sin(lx), where l is real. Here we have assumed l=2*pi."""
#####    psi = np.zeros(len(x))
#####    psi = (- np.sin(t)*np.sin(cotan(0.5*x + 0.25*np.pi)) + np.cos(t)*np.cos(cotan(0.5*x + 0.25*np.pi)))/(np.sin(x) + 1.) # !!! change 1+sinlx to 2 + sinlx and l from 1 to 2*pi # !!! change INCLUDE TIME and make 1+sinlx to 2 + sinlx and l from 1 to 2*pi
#####    return psi
#####    # !!! so for this function as we are just interested in the values at a full temporal revolution - can it just be the analytic_velocity_varying_space function? Because at one full revolution the cos(wt)=1 so it would be the same... 
#####



def velocity_varying_space(x, l=2.*np.pi):
    """This function returns a velocity field that is varying in space. The velocity field is given by u = 2 + sin(lx), where l is a real number."""
    u = 6. + 5.*np.sin(l*x) # currently 1+ not 2+!!!
    return u


def velocity_varying_space_time(x, t, l = 2.*np.pi, w = 2.*np.pi):
    """This function returns a velocity field that is varying in space and time. The velocity field is given by u = 2 + sin(lx) + sin(wt), where l and w are real numbers."""
    u = 1. + np.sin(l*x)*np.cos(w*t) # currently 1+ not 2+!!!
    return u


def analytic_constant(x, xmax, u=0., t=0., shift=0., ampl=1.):
    """Returns a constant value for the initial condition."""

    psi = np.full(len(x), 10.)

    return psi