"""This file includes various spatial (high-order) spatial discretisations in flux form.
Author: Amber te Winkel
Email: a.j.tewinkel@pgr.reading.ac.uk
Date: 23-02-2025
Initial note -> fourth - fifth in function names don't necessarily imply the order of accuracy in space. This is to be investigated further. As of 23-02-2025, the naming corresponds to the names in deriving_HO_spatialdiscs.ipynb.
"""

import numpy as np


def quadh(fm1, f, fp1): # Used for the quasicubic (third-order in space for uniform grid) interpolation. Was in schemes.py previously
    """This quadratic interpolation for f[i+1/2] leads to a cubic approximation when put in the FV ddx scheme. The quadratic interpolation matches the integral of the polynomial to the integral of the field over the three cells. See notes sent by James Kent on 28-11-2024.
    --- in ---
    fm1 : f[i-1]
    f   : f[i]
    fp1 : f[i+1]
    --- out --- 
    f[i+1/2] 
    """
    return (2.*fp1 + 5.*f - fm1)/6.


def QCmatrix(nx, dt, dx, u, alpha): # third-order in space for a uniform grid. Was in schemes.py previously
    """This function returns the matrix M for the quasi-cubic scheme. That is, the quasicubic approximation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!!!)
        M[i,i] = 1. + dt*alpha*(5.*np.roll(u,-1)[i] - 2.*u[i])/(6.*dx[i]) 
        M[i,i-1] = -dt*alpha*(np.roll(u,-1)[i] + 5.*u[i])/(6.*dx[i])
        M[i,(i-2)] = dt*alpha*u[i]/(6.*dx[i])   
        M[i,(i+1)%nx] = dt*alpha*np.roll(u,-1)[i]/(3.*dx[i])
    return M


def fourth22(field): # Was centred4() in schemes.py previously
    """Returns the flux for the centred fourth order spatial discretisation. Output defined at i-1/2."""
    return (-np.roll(field,-1) + 7.*field + 7.*np.roll(field,1) - np.roll(field,2))/12.


def Mfourth22(nx, dt, dx, u, alpha): # Was centred4matrix in schemes.py previously #!!! are the signs in this correct?
    """This function returns the matrix M for the fourth22 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-2)] = dt*alpha*u[i]/(12.*dx[i])
        M[i,i-1] = dt*alpha*(-np.roll(u,-1)[i] - 7.*u[i])/(12.*dx[i])
        M[i,i] = 1. + dt*alpha*(7.*np.roll(u,-1)[i] - 7.*u[i])/(12.*dx[i])
        M[i,(i+1)%nx] = dt*alpha*(7.*np.roll(u,-1)[i] + u[i])/(12.*dx[i])
        M[i,(i+2)%nx] = -dt*alpha*np.roll(u,-1)[i]/(12.*dx[i])
    return M


def fourth31(field):
    """Returns the flux for a spatial discretisation at [i] using [i+1], [i-1], [i-2], [i-3]. Output defined at i-1/2. Third-order in space."""
    return (11.*np.roll(field,1) + 11.*field + 5.*np.roll(field,2) - 3.*np.roll(field,3))/24.


def Mfourth31(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fourth31 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-3)] = 3.*dt*alpha*u[i]/(24.*dx[i])
        M[i,(i-2)] = dt*alpha*(-3.*np.roll(u,-1)[i] - 5.*u[i])/(24.*dx[i])
        M[i,i-1] = dt*alpha*(5.*np.roll(u,-1)[i] - 11.*u[i])/(24.*dx[i])
        M[i,i] = 1. + dt*alpha*(11.*np.roll(u,-1)[i] - 11.*u[i])/(24.*dx[i])
        M[i,(i+1)%nx] = 11.*dt*alpha*np.roll(u,-1)[i]/(24.*dx[i])
    return M


def fourth301(field):
    """Returns the flux for a spatial discretisation at [i] using [i+1], [i], [i-1], [i-2], [i-3]. Output defined at i-1/2."""
    return (3.*field + 13.*np.roll(field,1) - 5.*np.roll(field,2) + np.roll(field,3))/12.


def Mfourth301(nx, dt, dx, u, alpha): 
    """This function returns the matrix M for the fourth31 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-3)] = -dt*alpha*u[i]/(12.*dx[i])
        M[i,(i-2)] = dt*alpha*(np.roll(u,-1)[i] + 5.*u[i])/(12.*dx[i])
        M[i,i-1] = dt*alpha*(-5.*np.roll(u,-1)[i] - 13.*u[i])/(12.*dx[i])
        M[i,i] = 1. + dt*alpha*(13.*np.roll(u,-1)[i] - 3.*u[i])/(12.*dx[i])
        M[i,(i+1)%nx] = 3.*dt*alpha*np.roll(u,-1)[i]/(12.*dx[i])
    return M


def fifth302(field):
    """Returns the flux for a spatial discretisation at [i] using [i+2], [i+1], [i], [i-1], [i-2], [i-3]. Output defined at i-1/2."""
    return -1./20*np.roll(field,-1) + 9./20.*field + 47./60.*np.roll(field,1) - 13./60.*np.roll(field,2) + 1./30.*np.roll(field,3)


def Mfifth302(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fifth302 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-3)] = -1./30.*dt*alpha*u[i]/dx[i]
        M[i,(i-2)] = dt*alpha*(1./30.*np.roll(u,-1)[i] + 13./60.*u[i])/dx[i]
        M[i,i-1] = dt*alpha*(-13./60.*np.roll(u,-1)[i] - 47./60.*u[i])/dx[i]
        M[i,i] = 1. + dt*alpha*(47./60.*np.roll(u,-1)[i] - 9./20.*u[i])/dx[i]
        M[i,(i+1)%nx] = dt*alpha*(9./20.*np.roll(u,-1)[i] + 1./20.*u[i])/dx[i]
        M[i,(i+2)%nx] = -1./20.*dt*alpha*np.roll(u,-1)[i]/dx[i]
    return M


def fifth41(field):
    """Returns the flux for a spatial discretisation at [i] using [i+1], [i-1], [i-2], [i-3], [i-4]. Output defined at i-1/2."""
    return (5.*field + 5.*np.roll(field,1) + 7.*np.roll(field,2) - 7.*np.roll(field,3) + 2.*np.roll(field,4))/12.


def Mfifth41(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fifth41 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-4)] = -2.*dt*alpha*u[i]/(12.*dx[i])
        M[i,(i-3)] = dt*alpha*(2.*np.roll(u,-1)[i] + 7.*u[i])/(12.*dx[i])
        M[i,(i-2)] = dt*alpha*(-7.*np.roll(u,-1)[i] - 7.*u[i])/(12.*dx[i])
        M[i,i-1] = dt*alpha*(7.*np.roll(u,-1)[i] - 5.*u[i])/(12.*dx[i])
        M[i,i] = 1. + dt*alpha*(5.*np.roll(u,-1)[i] - 5.*u[i])/(12.*dx[i])
        M[i,(i+1)%nx] = 5.*dt*alpha*np.roll(u,-1)[i]/(12.*dx[i])
    return M


def fifth401(field): 
    """Returns the flux for a spatial discretisation at [i] using [i+1], [i], [i-1], [i-2], [i-3], [i-4]. Output defined at i-1/2."""
    return (1./5.*field + 77./60.*np.roll(field,1) - 43./60.*np.roll(field,2) + 17./60.*np.roll(field,3) - 1./20.*np.roll(field,4))


def Mfifth401(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fifth401 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-4)] = 1./20.*dt*alpha*u[i]/dx[i]
        M[i,(i-3)] = dt*alpha*(-1./20.*np.roll(u,-1)[i] - 17./60.*u[i])/dx[i]
        M[i,(i-2)] = dt*alpha*(17./60.*np.roll(u,-1)[i] + 43./60.*u[i])/dx[i]
        M[i,i-1] = dt*alpha*(-43./60.*np.roll(u,-1)[i] - 77./60.*u[i])/dx[i]
        M[i,i] = 1. + dt*alpha*(77./60.*np.roll(u,-1)[i] - 1./5.*u[i])/dx[i]
        M[i,(i+1)%nx] = 1./5.*dt*alpha*np.roll(u,-1)[i]/dx[i]
    return M


def fourth40(field):
    """Returns the flux for a spatial discretisation at [i] using [i], [i-1], [i-2], [i-3], [i-4]. Output defined at i-1/2."""
    return (25.*np.roll(field,1) - 23.*np.roll(field,2) + 13.*np.roll(field,3) - 3.*np.roll(field,4))/12.


def Mfourth40(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fourth40 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-4)] = 3.*dt*alpha*u[i]/(12.*dx[i])
        M[i,(i-3)] = dt*alpha*(-3.*np.roll(u,-1)[i] - 13.*u[i])/(12.*dx[i])
        M[i,(i-2)] = dt*alpha*(13.*np.roll(u,-1)[i] + 23.*u[i])/(12.*dx[i])
        M[i,i-1] = dt*alpha*(-23.*np.roll(u,-1)[i] - 25.*u[i])/(12.*dx[i])
        M[i,i] = 1. + 25.*dt*alpha*np.roll(u,-1)[i]/(12.*dx[i])
    return M


def fifth50(field):
    """Returns the flux for a spatial discretisation at [i] using [i], [i-1], [i-2], [i-3], [i-4], [i-5]. Output defined at i-1/2."""
    return (137.*np.roll(field,1) - 163.*np.roll(field,2) + 137.*np.roll(field,3) - 63.*np.roll(field,4) + 12.*np.roll(field,5))/60.


def Mfifth50(nx, dt, dx, u, alpha):
    """This function returns the matrix M for the fifth50 discretisation. That is, the discretisation at time level n+1, which is then on the LHS combined with the field[n+1,i] term."""
    M = np.zeros((nx, nx))
    for i in range(nx): # assumes u>0 # assumes A[0,0] = A[1,1] = A[2,2] (not always true!)
        M[i,(i-5)] = -12.*dt*alpha*u[i]/(60.*dx[i])
        M[i,(i-4)] = dt*alpha*(12.*np.roll(u,-1)[i] + 63.*u[i])/(60.*dx[i])
        M[i,(i-3)] = dt*alpha*(-63.*np.roll(u,-1)[i] - 137.*u[i])/(60.*dx[i])
        M[i,(i-2)] = dt*alpha*(137.*np.roll(u,-1)[i] + 163.*u[i])/(60.*dx[i])
        M[i,i-1] = dt*alpha*(-163.*np.roll(u,-1)[i] - 137.*u[i])/(60.*dx[i])
        M[i,i] = 1. + 137.*dt*alpha*np.roll(u,-1)[i]/(60.*dx[i])
    return M