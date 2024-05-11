# Module file with various experiments to analyze the schemes in schemes.py with
# Called in main.py
# Author:   Amber te Winkel
# Email:    a.j.tewinkel@pgr.reading.ac.uk


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
    rmse = np.sqrt((np.sum(dx*(field-analytic)*(field-analytic)))/(np.sum(dx*analytic*analytic) + 1e-16))

    return rmse


def totalmass(field, dx):
    """
    This function computes the total mass of the input field.
    --- Input ---
    field   : 1D array of floats, input field
    dx      : float, spatial discretisation
    --- Output ---
    TM      : float, total mass
    """
    TM = np.sum(field*dx)

    return TM


def check_conservation(init, field, dx):
    """
    This function computes the difference between the total mass of an initial field
    and a field later in time (diff = initial - later).
    --- Input ---
    init    : 1D array of floats, initial field
    field   : 1D array of floats, later field
    --- Output --- 
    diff    : float, difference
    """
    diff = totalmass(init, dx) - totalmass(field, dx)
    
    return diff


def check_boundedness(init, field):
    """
    This function checks boundedness for a field as compared to its initial condition. It returns a boolean,
    'True' if bounded, 'False' if unbounded. If unbounded, it also prints the bounds and the min/max value 
    that exceed these bounds.
    --- Input ---
    init    : 1D array of floats, initial condition that field has been time stepped from
    field   : 1D array of floats, solution field at a later point in time
    --- Output --- 
    bounded : boolean, True if field is bounded, False if unbounded
    """
    bounded = False
    minbound, maxbound = np.min(init), np.max(init)
    minfield, maxfield = np.min(field), np.max(field)
    if minfield < minbound or maxfield > maxbound:
        print('The below field is unbounded.')
        print(f'The (init) bounds are {minbound:.3f}, {maxbound:.3f}')
        print(f'The field has bounds: {minfield:.3f}, {maxfield:.3f}')
    else:
        bounded = True

    return bounded