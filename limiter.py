# This file defines the limiter used in schemes.py functions.
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk

import numpy as np
from numba_config import jitflags
from numba import njit

def FCT(field_LO, corr, dxc):
    """
    This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    field_LO: low-order solution
    corr: high-order correction, corr[i] is defined at i-1/2
    dxc: cell width
    previous: boolean to include previous time step field in max/min bounds as well (default=False)
    field: previous time step field (default=np.array([]))
    --- Output ---
    corrlim: limited high-order correction
    """
    n = len(field_LO)
    corrlim, C, amax, amin, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n),  np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    
    for i in range(n):
        if corr[i]*(field_LO[i] - field_LO[i-1]) < 0. and (corr[i]*(field_LO[(i+1)%n] - field_LO[i]) < 0. or corr[i]*(field_LO[i-1] - field_LO[i-2]) < 0.):
            corr[i] = 0.

        # Determine local max and min
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