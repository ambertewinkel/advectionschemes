# Module file to define the grid used in main.py
# Author:   Amber te Winkel
# Email:    ambertewinkel@gmail.com

import numpy as np

def coords_centralstretching(xmax, imax, i_maxC=0., dxcmin=0.):
    """
    This function implements a varying grid spacing, with stretching in the center: dx_center = 10*dx_boundary.
    We use a cosine function to define the grid spacing: dx[i] = dx0(6-5cos(2pi*(i-i_maxC)/imax)) for i ranging from 0 to imax.
    From integration we know that dx0 = xmax/(6*imax).
    We assume a periodic domain that ranges from 0 to xmax in size. xmax is not included in x-array
    --- Input:
    xmax    : float, domain size
    imax    : int, number of grid points
    i_maxC  : int, index where the Courant number is maximised/where dx is minimised for constant u throughout the domain
    dxcmin  : float, minimum dxc (connected to max Courant number allowed in the domain)
    --- Output:
    xf       : array of floats, spatial points of cell faces
    dxc      : array of floats, grid spacing between cell faces (dxc[i] = xf[i+1] - xf[i]), i.e., grid box size
    xc       : array of floats, spatial points of cell centers (xc[i+1] = 0.5*(xf[i+1] + xf[i]))
    dxf      : array of floats, grid spacing between cell centers (dxf[i] = xc[i+1] - xc[i])
    """
    # Initialisation
    xf, dxc, xc, dxf = np.zeros(imax), np.zeros(imax), np.zeros(imax), np.zeros(imax)
    
    # Setting grid values # !!! generalize in a separate function/have dxc as input to this function
    dx0 = xmax/(6*imax)
    dxc = [dx0*(6. - 5.*np.cos(2.*np.pi*(i-i_maxC)/imax)) for i in range(imax)] # define the grid spacing
    dxc = [dxc[i] if dxc[i] > dxcmin else dxcmin for i in range(imax)]

    # Calculating other grid quantities
    for i in range(len(dxc)-1):
        xf[i+1] = xf[i] + dxc[i]
    xc = 0.5*(np.roll(xf,-1) + xf)
    xc[-1] = 0.5*(xmax + xf[-1]) # periodic
    dxf = 0.5*(dxc + np.roll(dxc,-1))
   
    return  xf, dxc, xc, dxf

def coords_uniform(xmax, imax):
    """
    This function implements a uniform grid spacing.
    We assume a periodic domain that ranges from 0 to xmax in size. xmax is not included in x-array
    --- Input:
    xmax    : float, domain size
    imax    : int, number of grid points
    --- Output:
    xf       : array of floats, spatial points of cell faces
    dxc      : array of floats, grid spacing between cell faces (dxc[i] = xf[i+1] - xf[i]), i.e., grid box size
    xc       : array of floats, spatial points of cell centers (xc[i+1] = 0.5*(xf[i+1] + xf[i]))
    dxf      : array of floats, grid spacing between cell centers (dxf[i] = xc[i+1] - xc[i])
    """
    # Initialisation
    xf, dxc, xc, dxf = np.zeros(imax), np.zeros(imax), np.zeros(imax), np.zeros(imax)
    
    # Setting grid values
    dxc = np.full(imax, float(xmax/imax)) # define the grid spacing

    # Calculating other grid quantities
    for i in range(len(dxc)-1):
        xf[i+1] = xf[i] + dxc[i]
    xc = 0.5*(np.roll(xf,-1) + xf)
    xc[-1] = 0.5*(xmax + xf[-1]) # periodic
    dxf = dxc.copy()
   
    return  xf, dxc, xc, dxf

def cubLag(x_int, x, f):
    """
    Calculate the cubic Lagrange interpolation to a point.
    Input:
    x_int  : value of x to interpolate f to
    x   : array of four points x values
    f   : array of four points f values
    Output:
    f_int   : interpolated value
    """
    if len(x) != 4: print('Error: interpolation is not cubic Lagrange as expected.')
    xfrac = np.zeros(len(x))
    for i in range(len(x)):
        xother = np.delete(x, np.where(x == x[i]))
        xfrac[i] = np.prod((x_int - xother)/(x[i] - xother))
    f_int = np.dot(f, xfrac)
    return f_int