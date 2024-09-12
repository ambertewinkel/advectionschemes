# This file defines the limiter used in schemes.py functions.
# Author: Amber te Winkel
# Email: a.j.tewinkel@pgr.reading.ac.uk

import numpy as np


def FCT(field_LO, corr, dxc, previous=False, field=np.empty(0)):
    """This function limits the high-order correction based 
    on the low-order solution's local bounds. If previous=True, 
    then also the previous time step field can determine the max/min bounds.
    --- Input --- 
    ...
    corr[i] is defined at i-1/2
    """
    corrlim, C, amax, amin, fieldmax, fieldmin, Pp, Qp, Rp, Pm, Qm, Rm = np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)),  np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO)), np.zeros(len(field_LO))
    
    for i in range(len(field_LO)):
        if corr[i+1]*(field_LO[i+1] - field_LO[i]) < 0. and (corr[i+1]*(field_LO[i+2] - field_LO[i+1]) < 0. or corr[i+1]*(field_LO[i] - field_LO[i-1]) < 0.):
            corrlim[i] = 0.
        else:
            # add option to also have max/min from previous field
            if previous == True:
                amax[i] = max(field_LO[i], field[i])
                amin[i] = min(field_LO[i], field[i])
            else:
                amax[i] = field_LO[i] # do these need to be copied?
                amin[i] = field[i]
            
            # Determine local max and min
            fieldmax[i] = max(field_LO[i-1], field_LO[i], field_LO[i+1])
            fieldmin[i] = min(field_LO[i-1], field_LO[i], field_LO[i+1])

            Pp[i] = max(0., corr[i]) - min(0., corr[i+1])
            Qp[i] = (fieldmax[i] - field_LO[i])*dxc[i]
            Rp[i] = min(1., Qp[i]/Pp[i]) if Pp[i] > 0. else 0.
            
            Pm[i] = max(0., corr[i+1]) - min(0., corr[i-1])
            Qm[i] = (field_LO[i] - fieldmin[i])*dxc[i]
            Rm[i] = min(1., Qm[i]/Pm[i]) if Pm[i] > 0. else 0.

            # Determine C at face i-1/2
            C[i] = min(Rp[i-1], Rm[i]) if corr[i] < 0. else min(Rp[i], Rm[i-1])

            # Determine limited correction
            corrlim[i] = C[i]*corr[i] 

    return corrlim