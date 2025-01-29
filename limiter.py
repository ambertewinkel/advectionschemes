# This file defines the limiter used in schemes.py functions.
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk

import numpy as np
from numba_config import jitflags
from numba import njit
import matplotlib.pyplot as plt

def FCT(field_LO, corr, dxc, previous, double=False, secondfield=None):
    """
    This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    field_LO: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2
    dxc: cell width
    previous: previous time step field - if an element in previous is None, it is not used.
    double: double FCT limiter, True if FCT is applied twice, see doubleFCT function below.
    fieldlim: limited high-order field after one FCT application, only used if double==True.
    --- Output ---
    corrlim: limited high-order correction
    """
    n = len(field_LO)
    corrlim, C, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n), np.zeros(n), np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if corr[i]*(field_LO[i] - field_LO[i-1]) <= 0. and (corr[i]*(field_LO[(i+1)%n] - field_LO[i]) <= 0. or corr[i]*(field_LO[i-1] - field_LO[i-2]) <= 0.):
            corr[i] = 0.

        # Determine local max and min
        if previous[i] is not None and double == False:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n]])
        elif previous[i] is not None and double == True:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], previous[i-1], previous[i], previous[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
        elif previous[i] is None and double == True:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n], secondfield[i-1], secondfield[i], secondfield[(i+1)%n]])
        else:
            fieldmax[i] = max([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])
            fieldmin[i] = min([field_LO[i-1], field_LO[i], field_LO[(i+1)%n]])

        Pp[i] = max([0., corr[i]]) - min([0., corr[(i+1)%n]])
        Qp[i] = (fieldmax[i] - field_LO[i])*dxc[i]
        Rp[i] = min([1., Qp[i]/Pp[i]]) if Pp[i] > 0. else 0.
        
        Pm[i] = max([0., corr[(i+1)%n]]) - min([0., corr[i]])
        Qm[i] = (field_LO[i] - fieldmin[i])*dxc[i]
        Rm[i] = min([1., Qm[i]/Pm[i]]) if Pm[i] > 0. else 0.

    for i in range(n):
        # Determine C at face i-1/2
        C[i] = min([Rp[i-1], Rm[i]]) if corr[i] < 0. else min([Rp[i], Rm[i-1]])

        # Determine limited correction
        corrlim[i] = C[i]*corr[i]

    return corrlim


def nonneg(field, flx, dxc): # assumes constant dxc
    """
    This function implements nonnegativity as discussed with HW in the 27-01-2025 meeting.
    field[i] defined at i, resulting field from the fluxes defined by flx
    flx[i] defined at i-1/2, the flux that leads to field being as it is
    """
    for i in range(len(field)):
        while field[i] < 0. and flx[i] < 0.:
            addflx = -field[i]*dxc[i] # field[i] = negative, addflx = positive
            flx[i] += addflx 
            field[i] += addflx/dxc[i] # adjust field[i]
            field[i-1] -= addflx/dxc[i] # adjust field[i-1] # assumes constant dxc
            i = (i - 1)%len(field) # adjust field[i-1] and check in the same loop whether it has become negative, if so limit flx[i-2] and so on until no negative values are found
        while field[i] < 0. and flx[(i + 1)%len(field)] > 0.:
            rmvflx = field[i]*dxc[i] # field[i] = negative, rmvflx = negative
            flx[(i + 1)%len(field)] += rmvflx 
            field[i] -= rmvflx/dxc[i] # adjust field[i]
            field[(i+1)%len(field)] += rmvflx/dxc[i] # adjust field[i+1] # assumes constant dxc
            i = (i + 1)%len(field) # adjust field[i+1] and check in the same loop whether it has become negative, if so limit flx[i+2] and so on until no negative values are found       

    return field


def doubleFCT(field_LO, corr, dxc, previous):
    """
    This function implements double FCT as discussed with HW in the 27-01-2025 meeting.
    """
    corrlim = FCT(field_LO, corr, dxc, previous) # first FCT application
    fieldlim = field_LO - (np.roll(corrlim,-1) - corrlim)/dxc # HO bounded solution after one FCT application
    newcorr = corr - corrlim
    newcorrlim = FCT(fieldlim, newcorr, dxc, previous, double=True, secondfield=field_LO) # second FCT application
    
    return corrlim + newcorrlim