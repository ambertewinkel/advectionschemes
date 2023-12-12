# Module file with various experiments to analyze the schemes in schemes.py with
# Called in main.py

import numpy as np
import matplotlib.pyplot as plt
 
def totalvariation(field):
    """
    This function computes the total variation of a periodic input field.
    --- Input ---
    field   : 1D array, assumed to be periodic
    --- Output ---
    TV      : total variation, i.e., the sum over all grid points of the
              absolute value of the change from that point to the next
    """

    TV = 0.0
    for i in range(len(field)-1):
        TV += abs(field[i+1] - field[i])
    TV += abs(field[0] - field[-1])
    return TV

def rmse(field, analytic, dx):
    """
    This function calculates the root-mean-square-error (RMSE) for a finite difference (FD)
    scheme as compared to the analytic solution. Assumed is a 1D spatial input for a
    certain point in time t.
    --- Input --- 
    field       : 1D array of floats, FD spatial field at t
    analytic    : analytic solution at t
    dx          : spatial discretisation
    --- Output --- 
    rmse     : 1D array of floats, root-mean-square-error.
    """
    rmse = np.zeros(len(field))
    rmse = np.sqrt((np.sum(dx*(field-analytic)*(field-analytic)))/(np.sum(dx*analytic*analytic)))

    return rmse